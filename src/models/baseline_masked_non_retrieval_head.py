import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class BaselineMaskedNonRetrievalHead(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self.num_layers = len(self.model.model.layers)

        self._load_retrieval_heads()
        self.num_retrieval_heads = self.decoder_configs.configs.num_retrieval_heads
        assert (
            self.num_retrieval_heads < 0
        ), "Number of retrieval heads should be negative"  # negative number of retrieval heads to signify selecting random heads
        self.random_heads = self._construct_random_head(-self.num_retrieval_heads)
        print("Random heads: ", self.random_heads)

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

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        return self._generate(
            inputs, return_attentions=return_attentions, block_list=self.random_heads
        )

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

            outputs = self.model(input_ids, block_list=self.random_heads)[0]
            outputs = outputs.squeeze(0).log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs
