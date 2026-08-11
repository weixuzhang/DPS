from typing import List, Optional, Tuple

import copy
import torch
from transformers import AutoTokenizer

from src.utils.modelling_llama import LlamaForCausalLM
from src.utils.modelling_mistral import MistralForCausalLM
from src.utils.modelling_qwen2 import Qwen2ForCausalLM

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class ContrastiveDecoding(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        if "llama" in decoder_configs.configs.amateur_model_name_or_path.lower():
            self.amateur_model = LlamaForCausalLM.from_pretrained(
                decoder_configs.configs.amateur_model_name_or_path,
                use_flash_attention_2="flash_attention_2",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
        elif "mistral" in decoder_configs.configs.amateur_model_name_or_path.lower():
            self.amateur_model = MistralForCausalLM.from_pretrained(
                decoder_configs.configs.amateur_model_name_or_path,
                use_flash_attention_2="flash_attention_2",
                attn_implementation="flash_attention_2",
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            ).eval()
        elif "qwen2" in decoder_configs.configs.amateur_model_name_or_path.lower():
            self.amateur_model = Qwen2ForCausalLM.from_pretrained(
                decoder_configs.configs.amateur_model_name_or_path,
                use_flash_attention_2="flash_attention_2",
                attn_implementation="flash_attention_2",
                torch_dtype="auto",
                device_map="auto",
            ).eval()

        self.amateur_tokenizer = AutoTokenizer.from_pretrained(
            decoder_configs.configs.amateur_model_name_or_path
        )

        self.alpha = self.decoder_configs.configs.alpha

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        assert (
            not return_attentions
        ), "Return attentions not supported for DeCoReEntropy"
        self.model.eval()

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
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)
                expert_lm_output = self.model(
                    input_ids=last_input_token,
                    past_key_values=expert_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )

                # Forward pass
                amateur_lm_output = self.amateur_model(
                    input_ids=last_input_token,
                    past_key_values=amateur_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )

                expert_past_kv = expert_lm_output.past_key_values
                amateur_past_kv = amateur_lm_output.past_key_values

                expert_logits = expert_lm_output.logits[0, -1].log_softmax(dim=-1)
                amateur_logits = amateur_lm_output.logits[0, -1].log_softmax(dim=-1)

                # Contrast expert LM and amateur LM scores
                next_token_logits = expert_logits - self.alpha * amateur_logits

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break

            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "attentions": {}, "alphas": []}

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

            lm_output = self.model(input_ids, attn_mode=self.attn_mode)[0]
            amateur_output = self.amateur_model(
                amateur_input_ids, attn_mode=self.attn_mode
            )[0]

            base_logits = lm_output[0, prefix_ids.shape[-1] - 1 : -1, :]
            amateur_logits = amateur_output[0, amateur_prefix_ids.shape[-1] - 1 : -1, :]

            base_logits = base_logits.log_softmax(dim=-1)
            amateur_logits = amateur_logits.log_softmax(dim=-1)

            diff_logits = base_logits - self.alpha * amateur_logits

            if self.decoder_configs.configs.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
