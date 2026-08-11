from typing import List, Optional, Tuple

import copy
import random
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class DeCoReRandomEntropy(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self.num_layers = len(self.model.model.layers)

        self._load_retrieval_heads()
        print("Retrieval heads: ", self.retrieval_heads)
        self.num_retrieval_heads = self.decoder_configs.configs.num_retrieval_heads
        assert (
            self.num_retrieval_heads < 0
        ), "Number of retrieval heads should be negative"  # negative number of retrieval heads to signify selecting random heads
        self.random_heads = self._construct_random_head(-self.num_retrieval_heads)
        print("Random heads: ", self.random_heads)

        self.alpha_cap = decoder_configs.configs.get("alpha_cap", None)

        self.scale_alpha = decoder_configs.configs.get("scale_alpha", False)

    def _load_retrieval_heads(self):
        model_base_name = self.model_configs.configs.model_name_or_path.split("/")[1]

        with open(
            os.path.join(
                self.decoder_configs.configs.retrieval_heads_dir,
                f"{model_base_name}.json",
            )
        ) as file:
            head_list = json.loads(file.readline())

        stable_block_list = [(l[0], np.mean(l[1])) for l in head_list.items()]
        stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
        self.retrieval_heads = [
            [int(ll) for ll in l[0].split("-")] for l in stable_block_list
        ][:100]

    def _construct_random_head(self, n):
        results = []
        seed_list = [i for i in range(self.num_layers)]
        random.shuffle(seed_list)
        while len(results) < n:
            l, h = random.choices(seed_list, k=2)
            if (l, h) in results or (l, h) in self.retrieval_heads:
                continue
            else:
                results.append((l, h))
        return results

    def _calculate_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        if self.scale_alpha:
            entropy = entropy / np.log(probs.shape[-1])

        return entropy

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        assert (
            not return_attentions
        ), "Return attentions not supported for DeCoReVanilla"
        self.model.eval()

        prompt = inputs["prompted_question"][0]
        inputs = self._verbalise_input(prompt).to(self.model.device)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            last_input_token = inputs[:, -1]
            base_past_kv = copy.deepcopy(input_logits.past_key_values)
            hallucinated_past_kv = copy.deepcopy(input_logits.past_key_values)
            alphas = []
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                base_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=base_past_kv,
                    use_cache=True,
                    attn_mode="torch",
                )
                hallucinated_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=hallucinated_past_kv,
                    use_cache=True,
                    attn_mode="torch",
                    block_list=self.random_heads,
                )

                base_past_kv = base_outputs.past_key_values
                hallucinated_past_kv = hallucinated_outputs.past_key_values

                alpha = self._calculate_entropy(base_outputs.logits[0, -1])

                alphas += [alpha.item()]

                if self.alpha_cap:
                    # If the entropy is too high, cap the alpha with the entropy cap
                    alpha = torch.min(
                        alpha, torch.tensor(self.alpha_cap).to(alpha.device)
                    )

                next_token_logits = (1 + alpha) * base_outputs.logits[
                    0, -1
                ] - alpha * hallucinated_outputs.logits[0, -1]

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "attentions": {}, "alphas": alphas}

    def lm_score(
        self,
        prompt,
        answer,
    ):
        prompted_question = prompt["prompted_question"][0]

        # Only relevant for instruct model
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

            base_outputs = self.model(input_ids)[0]
            hallucinated_outputs = self.model(input_ids, block_list=self.random_heads)[
                0
            ]

            base_logits = base_outputs[0, prefix_ids.shape[-1] - 1 : -1, :]
            hallucinated_logits = hallucinated_outputs[
                0, prefix_ids.shape[-1] - 1 : -1, :
            ]

            entropies = []
            for i in range(base_logits.shape[0]):
                entropies += [self._calculate_entropy(base_logits[i, :])]
            alpha = torch.stack(entropies).unsqueeze(1)

            if self.alpha_cap:
                # If the entropy is too high, cap the alpha with the entropy cap
                alpha = torch.min(alpha, torch.tensor(self.alpha_cap).to(alpha.device))

            base_logits = base_logits.log_softmax(dim=-1)
            hallucinated_logits = hallucinated_logits.log_softmax(dim=-1)

            diff_logits = (1 + alpha) * base_logits - alpha * hallucinated_logits

            if self.decoder_configs.configs.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
