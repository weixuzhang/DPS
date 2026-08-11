"""
Differential Preference Steering (DPS)

This method steers language model generation based on detected preference heads.
Similar to DeCoRe but uses preference heads instead of retrieval heads to
personalize model outputs according to user preferences.
"""

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


class DPS(BaseModel):
    """Differential Preference Steering decoder.
    
    Uses detected preference heads to steer generation towards user preferences
    by amplifying activations in preference-relevant attention heads.
    """
    
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

            self._load_preference_heads(
                decoder_configs.configs.amateur_model_name_or_path,
                decoder_configs.configs.task
            )
        else:
            self.amateur_model = None
            self._load_preference_heads(
                model_configs.configs.model_name_or_path,
                decoder_configs.configs.task
            )

        print(f"Preference heads for {decoder_configs.configs.task}: ", self.preference_heads)

        self.alpha_cap = decoder_configs.configs.get("alpha_cap", None)

        self.scale_alpha = decoder_configs.configs.get("scale_alpha", False)

    def _load_preference_heads(self, model_name_or_path, task):
        """Load preference heads detected for the specific task.
        
        Args:
            model_name_or_path: Model identifier (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
            task: Task name (e.g., "LaMP-1", "LaMP-2")
        """
        print(f"Loading preference heads for {model_name_or_path} on {task}")
        self.num_preference_heads = self.decoder_configs.configs.num_preference_heads

        # Extract model base name
        model_base_name = model_name_or_path.split("/")[-1]  # e.g., "Meta-Llama-3-8B-Instruct"
        
        # Map task name from LAMP_X to LaMP-X format if needed
        if task.startswith("LAMP_"):
            task = task.replace("LAMP_", "LaMP-")
        
        # Construct preference heads file path
        preference_heads_file = os.path.join(
            self.decoder_configs.configs.preference_heads_dir,
            f"{model_base_name}_{task.replace('-', '_')}_top_heads.json"
        )
        
        print(f"Loading preference heads from: {preference_heads_file}")
        
        if not os.path.exists(preference_heads_file):
            raise FileNotFoundError(
                f"Preference heads file not found: {preference_heads_file}\n"
                f"Please run preference head detection first:\n"
                f"  cd <PROJECT_ROOT>/preference_head\n"
                f"  python preference_head_detection.py --task {task}"
            )
        
        with open(preference_heads_file, 'r') as f:
            preference_data = json.load(f)
        
        # Extract preference heads (list of [layer, head] pairs)
        all_preference_heads = preference_data['preference_heads']
        
        # Select top N preference heads
        self.preference_heads = all_preference_heads[:self.num_preference_heads]
        
        print(f"Loaded {len(self.preference_heads)} preference heads (requested {self.num_preference_heads})")

    def _calculate_entropy(self, logits):
        """Calculate entropy for preference steering strength."""
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        if self.scale_alpha:
            entropy = entropy / np.log(probs.shape[-1])

        return entropy

    def generate_self_contrast(self, inputs, return_attentions: bool = False) -> dict:
        """Generate with preference steering.
        
        Uses detected preference heads to guide generation towards user preferences.
        Similar to DeCoRe: compares base model vs model with preference heads blocked.
        """
        assert (
            not return_attentions
        ), "Return attentions not supported for DPS"
        self.model.eval()

        prompt = inputs["prompted_question"][0]

        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        # Predict with preference steering (like DeCoRe's self-contrast)
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            last_input_token = tokenised_inputs[:, -1]
            base_past_kv = copy.deepcopy(input_logits.past_key_values)
            depersonalized_past_kv = copy.deepcopy(input_logits.past_key_values)
            alphas = []
            
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                # Base model (with all heads including preference heads)
                base_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=base_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )
                
                # Depersonalized model (preference heads blocked)
                depersonalized_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=depersonalized_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                    block_list=self.preference_heads,  # Block preference heads
                )

                base_past_kv = base_outputs.past_key_values
                depersonalized_past_kv = depersonalized_outputs.past_key_values

                # Calculate entropy for steering strength
                alpha = self._calculate_entropy(base_outputs.logits[0, -1])
                alphas += [alpha.item()]

                if self.alpha_cap:
                    # If the entropy is too high, cap the alpha
                    alpha = torch.min(
                        alpha, torch.tensor(self.alpha_cap).to(alpha.device)
                    )

                # Contrastive decoding: amplify base, suppress depersonalized
                base_logits = base_outputs.logits[0, -1]
                base_logits = base_logits.log_softmax(dim=-1)
                depersonalized_logits = depersonalized_outputs.logits[0, -1]
                depersonalized_logits = depersonalized_logits.log_softmax(dim=-1)

                # DPS formula (same as DeCoRe): (1 + α) * base - α * depersonalized
                next_token_logits = (
                    1 + alpha
                ) * base_logits - alpha * depersonalized_logits

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
                    
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {
            "decoded_text": decoded_text,
            "alphas": alphas,
            "attentions": {}
        }


    def generate(
        self,
        inputs: dict,
        return_attentions: bool = False,
    ) -> dict:
        """Generate text with preference steering.
        
        Args:
            inputs: Dictionary containing "prompted_question" and optionally "verbalised_instruction"
            return_attentions: Whether to return attention weights (not supported)
            
        Returns:
            Dictionary with "decoded_text", "alphas", and "attentions" keys
        """
        return self.generate_self_contrast(inputs, return_attentions)

    def lm_score(
        self,
        prompt,
        answer,
    ):
        """Calculate log probability score with preference steering.
        
        Uses contrastive decoding like DeCoRe but with preference heads.
        Compares base model vs model with preference heads blocked.
        
        Args:
            prompt: Dictionary containing "prompted_question" and "verbalised_instruction"
            answer: The answer text to score
            
        Returns:
            Log probability score with preference steering (float)
        """
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

            # Base model outputs (with preference heads)
            base_outputs = self.model(input_ids, attn_mode=self.attn_mode)[0]
            
            # Depersonalized outputs (preference heads blocked)
            depersonalized_outputs = self.model(
                input_ids, block_list=self.preference_heads, attn_mode=self.attn_mode
            )[0]

            base_logits = base_outputs[0, prefix_ids.shape[-1] - 1 : -1, :]
            depersonalized_logits = depersonalized_outputs[
                0, prefix_ids.shape[-1] - 1 : -1, :
            ]

            # Calculate entropy for each position
            entropies = []
            for i in range(base_logits.shape[0]):
                entropies += [self._calculate_entropy(base_logits[i, :])]

            alpha = torch.stack(entropies).unsqueeze(1)

            if self.alpha_cap:
                # If the entropy is too high, cap the alpha
                alpha = torch.min(alpha, torch.tensor(self.alpha_cap).to(alpha.device))

            base_logits = base_logits.log_softmax(dim=-1)
            depersonalized_logits = depersonalized_logits.log_softmax(dim=-1)

            # DPS contrastive formula
            diff_logits = (1 + alpha) * base_logits - alpha * depersonalized_logits

            if self.decoder_configs.configs.post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
