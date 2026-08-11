from typing import List, Optional, Tuple

import copy
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel
from src.utils.modelling_llama import LlamaForCausalLM


class DeCoReEntropyGain(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        if decoder_configs.configs.amateur_model_name_or_path is not None:
            if "llama" in decoder_configs.configs.amateur_model_name_or_path.lower():
                self.amateur_model = LlamaForCausalLM.from_pretrained(
                    decoder_configs.configs.amateur_model_name_or_path,
                    use_flash_attention_2="flash_attention_2",
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                ).eval()
                self.amateur_attn_mode = "flash"
            else:
                raise NotImplementedError(
                    "Amateur model other than Llama3-8b-Instruct is not supported yet"
                )

            self.amateur_tokenizer = AutoTokenizer.from_pretrained(
                decoder_configs.configs.amateur_model_name_or_path
            )

            self._load_retrieval_heads(
                decoder_configs.configs.amateur_model_name_or_path
            )
        else:
            self.amateur_model = None
            self._load_retrieval_heads(model_configs.configs.model_name_or_path)

        print("Retrieval heads: ", self.retrieval_heads)

        self.alpha_cap = decoder_configs.configs.get("alpha_cap", None)

        self.scale_alpha = decoder_configs.configs.get("scale_alpha", False)

    def _load_retrieval_heads(self, model_name_or_path):
        print(f"Loading retrieval heads {model_name_or_path}")
        self.num_retrieval_heads = self.decoder_configs.configs.num_retrieval_heads

        model_base_name = model_name_or_path.split("/")[1]

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
        ][: self.num_retrieval_heads]

    def _calculate_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        if self.scale_alpha:
            entropy = entropy / np.log(probs.shape[-1])

        return entropy

    def generate_self_contrast(self, inputs, return_attentions: bool = False) -> dict:
        assert (
            not return_attentions
        ), "Return attentions not supported for DeCoReEntropy"
        self.model.eval()

        prompt = inputs["prompted_question"][0]

        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            last_input_token = tokenised_inputs[:, -1]
            base_past_kv = copy.deepcopy(input_logits.past_key_values)
            hallucinated_past_kv = copy.deepcopy(input_logits.past_key_values)
            alphas = []
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                base_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=base_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )
                hallucinated_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=hallucinated_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                    block_list=self.retrieval_heads,
                )

                base_past_kv = base_outputs.past_key_values
                hallucinated_past_kv = hallucinated_outputs.past_key_values

                base_entropy = self._calculate_entropy(base_outputs.logits[0, -1])
                hallucinated_entropy = self._calculate_entropy(
                    hallucinated_outputs.logits[0, -1]
                )

                alpha = hallucinated_entropy - base_entropy

                alphas += [alpha.item()]

                if self.alpha_cap:
                    # If the entropy is too high, cap the alpha with the entropy cap
                    alpha = torch.min(
                        alpha, torch.tensor(self.alpha_cap).to(alpha.device)
                    )

                base_logits = base_outputs.logits[0, -1]
                base_logits = base_logits.log_softmax(dim=-1)
                hallucinated_logits = hallucinated_outputs.logits[0, -1]
                hallucinated_logits = hallucinated_logits.log_softmax(dim=-1)

                next_token_logits = (
                    1 + alpha
                ) * base_logits - alpha * hallucinated_logits

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "attentions": {}, "alphas": alphas}

    def generate_amateur_contrast(
        self, inputs, return_attentions: bool = False
    ) -> dict:
        assert (
            not return_attentions
        ), "Return attentions not supported for DeCoReEntropy"

        assert self.amateur_model is not None, "Amateur model not loaded"

        self.model.eval()
        self.amateur_model.eval()

        prompt = inputs["prompted_question"][0]

        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        expert_tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        amateur_tokenised_inputs = self._verbalise_input(
            prompt,
            use_system_prompt=use_system_prompt,
            tokenizer=self.amateur_tokenizer,
        ).to(self.model.device)

        # Predict
        with torch.inference_mode():
            expert_input_logits = self.model(
                input_ids=expert_tokenised_inputs[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            amateur_input_logits = self.amateur_model(
                input_ids=amateur_tokenised_inputs[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            generated_ids = []
            last_input_token = expert_tokenised_inputs[:, -1]
            expert_past_kv = copy.deepcopy(expert_input_logits.past_key_values)
            amateur_past_kv = copy.deepcopy(amateur_input_logits.past_key_values)
            alphas = []
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                expert_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=expert_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )
                amateur_outputs = self.amateur_model(
                    input_ids=last_input_token,
                    past_key_values=amateur_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                    block_list=self.retrieval_heads,
                )

                expert_past_kv = expert_outputs.past_key_values
                amateur_past_kv = amateur_outputs.past_key_values

                base_entropy = self._calculate_entropy(expert_outputs.logits[0, -1])
                hallucinated_entropy = self._calculate_entropy(
                    amateur_outputs.logits[0, -1]
                )

                alpha = hallucinated_entropy - base_entropy

                alphas += [alpha.item()]

                if self.alpha_cap:
                    # If the entropy is too high, cap the alpha with the entropy cap
                    alpha = torch.min(
                        alpha, torch.tensor(self.alpha_cap).to(alpha.device)
                    )

                expert_logits = expert_outputs.logits[0, -1]
                expert_logits = expert_logits.log_softmax(dim=-1)
                amateur_logits = amateur_outputs.logits[0, -1]
                amateur_logits = amateur_logits.log_softmax(dim=-1)

                next_token_logits = (1 + alpha) * expert_logits - alpha * amateur_logits

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "attentions": {}, "alphas": alphas}

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        if self.amateur_model is not None:
            return self.generate_amateur_contrast(inputs, return_attentions)
        else:
            return self.generate_self_contrast(inputs, return_attentions)

    def lm_score_self_contrast(
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

            base_outputs = self.model(input_ids, attn_mode=self.attn_mode)[0]
            hallucinated_outputs = self.model(
                input_ids, block_list=self.retrieval_heads, attn_mode=self.attn_mode
            )[0]

            base_logits = base_outputs[0, prefix_ids.shape[-1] - 1 : -1, :]
            hallucinated_logits = hallucinated_outputs[
                0, prefix_ids.shape[-1] - 1 : -1, :
            ]

            entropies = []
            for i in range(base_logits.shape[0]):
                base_entropy = self._calculate_entropy(base_logits[i, :])
                hallucinated_entropy = self._calculate_entropy(
                    hallucinated_logits[i, :]
                )
                entropies += [hallucinated_entropy - base_entropy]

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

    def lm_score_amateur_contrast(
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

            expert_input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
            ).to(self.model.device)
            expert_prefix_ids = self._verbalise_input(
                prompted_question, use_system_prompt=use_system_prompt
            ).to(self.model.device)
            continue_ids = expert_input_ids[0, expert_prefix_ids.shape[-1] :]

            amateur_input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
                tokenizer=self.amateur_tokenizer,
            ).to(self.amateur_model.device)
            amateur_prefix_ids = self._verbalise_input(
                prompted_question,
                use_system_prompt=use_system_prompt,
                tokenizer=self.amateur_tokenizer,
            ).to(self.amateur_model.device)

            expert_outputs = self.model(expert_input_ids, attn_mode=self.attn_mode)[0]
            amateur_outputs = self.amateur_model(
                amateur_input_ids,
                block_list=self.retrieval_heads,
                attn_mode=self.amateur_attn_mode,
            )[0]

            expert_logits = expert_outputs[0, expert_prefix_ids.shape[-1] - 1 : -1, :]
            amateur_logits = amateur_outputs[
                0, amateur_prefix_ids.shape[-1] - 1 : -1, :
            ]

            entropies = []
            for i in range(expert_logits.shape[0]):
                expert_entropy = self._calculate_entropy(expert_logits[i, :])
                hallucinated_entropy = self._calculate_entropy(amateur_logits[i, :])
                entropies += [hallucinated_entropy - expert_entropy]

            alpha = torch.stack(entropies).unsqueeze(1)

            if self.alpha_cap:
                # If the entropy is too high, cap the alpha with the entropy cap
                alpha = torch.min(alpha, torch.tensor(self.alpha_cap).to(alpha.device))

            expert_logits = expert_logits.log_softmax(dim=-1)
            amateur_logits = amateur_logits.log_softmax(dim=-1)

            diff_logits = (1 + alpha) * expert_logits - alpha * amateur_logits

            if self.decoder_configs.configs.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs

    def lm_score(
        self,
        prompt,
        answer,
    ):
        if self.amateur_model is not None:
            return self.lm_score_amateur_contrast(prompt, answer)
        else:
            return self.lm_score_self_contrast(prompt, answer)
