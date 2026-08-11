from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class DoLa(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self.dola_layers = self.decoder_configs.configs.dola_layers

        self.post_softmax = self.decoder_configs.configs.post_softmax

        self.num_layers = len(self.model.model.layers)
        mid_point = self.num_layers // 2

        if self.dola_layers == "low":
            self.candidate_premature_layers = list(range(0, mid_point, 2)) + [
                self.num_layers
            ]
        elif self.dola_layers == "high":
            self.candidate_premature_layers = list(
                range(mid_point, self.num_layers, 2)
            ) + [self.num_layers]
        self.mature_layer = self.candidate_premature_layers[-1]

    def _calculate_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        return entropy

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        self.model.eval()

        prompt = inputs["prompted_question"][0]

        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                tokenised_inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                output_logits=True,
                dola_layers=self.dola_layers,
                return_dict_in_generate=True,
            )
            decoded_text = self.tokenizer.decode(
                outputs.sequences[0, tokenised_inputs.size(1) :],
                skip_special_tokens=True,
            )
        logits = torch.stack(outputs.logits, dim=1)

        entropies = self._calculate_entropy(logits)
        entropies = entropies.cpu().numpy().tolist()

        return {"decoded_text": decoded_text, "alphas": entropies, "attentions": {}}

    def lm_score(
        self,
        prompt,
        answer,
    ):
        prompted_question = prompt["prompted_question"][0]

        if len(prompt["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        with torch.no_grad():
            if type(prompted_question) == list:
                input_text = prompted_question + [answer]
            else:
                input_text = prompted_question + answer
            input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
            ).to(self.model.device)
            prefix_ids = self._verbalise_input(
                prompted_question, use_system_prompt=use_system_prompt
            ).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            premature_layer_dist = {l: 0 for l in self.candidate_premature_layers}
            picked_logits = []
            result_dict = {}
            premature_layers = []

            dict_outputs, outputs = self.model(
                input_ids=input_ids,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                early_exit_layers=self.candidate_premature_layers + [self.mature_layer],
                attn_mode=self.attn_mode,
            )

            for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                # Pick the less like layer to contrast with
                # 1. Stacking all premature_layers into a new dimension
                stacked_premature_layers = torch.stack(
                    [
                        dict_outputs[i][:, seq_i, :]
                        for i in self.candidate_premature_layers
                    ],
                    dim=0,
                )

                # 2. Calculate the softmax values for mature_layer and all premature_layers
                softmax_mature_layer = F.softmax(
                    dict_outputs[self.mature_layer][:, seq_i, :], dim=-1
                )  # shape: (batch_size, num_features)
                softmax_premature_layers = F.softmax(
                    stacked_premature_layers, dim=-1
                )  # shape: (num_premature_layers, batch_size, num_features)

                # 3. Calculate M, the average distribution
                M = 0.5 * (
                    softmax_mature_layer[None, :, :] + softmax_premature_layers
                )  # shape: (num_premature_layers, batch_size, num_features)

                # 4. Calculate log-softmax for the KL divergence
                log_softmax_mature_layer = F.log_softmax(
                    dict_outputs[self.mature_layer][:, seq_i, :], dim=-1
                )  # shape: (batch_size, num_features)
                log_softmax_premature_layers = F.log_softmax(
                    stacked_premature_layers, dim=-1
                )  # shape: (num_premature_layers, batch_size, num_features)

                # 5. Calculate the KL divergences and then the JS divergences
                kl1 = F.kl_div(
                    log_softmax_mature_layer[None, :, :], M, reduction="none"
                ).mean(
                    -1
                )  # shape: (num_premature_layers, batch_size)
                kl2 = F.kl_div(log_softmax_premature_layers, M, reduction="none").mean(
                    -1
                )  # shape: (num_premature_layers, batch_size)
                js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                # 6. Reduce the batchmean
                js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                premature_layer = self.candidate_premature_layers[
                    int(js_divs.argmax().cpu().item())
                ]
                premature_layer_dist[premature_layer] += 1

                premature_layers.append(premature_layer)

            base_logits = torch.zeros_like(
                dict_outputs[self.mature_layer][0, prefix_ids.shape[-1] - 1 : -1]
            )
            for i, l in enumerate(premature_layers):
                base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
            final_logits = dict_outputs[self.mature_layer][
                0, prefix_ids.shape[-1] - 1 : -1
            ]
            final_logits = final_logits.log_softmax(dim=-1)
            base_logits = base_logits.log_softmax(dim=-1)
            diff_logits = final_logits - base_logits
            if self.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
