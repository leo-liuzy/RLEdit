from typing import Dict, List

import numpy as np
import scipy

import torch
import random
import unicodedata

from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
# from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
# from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from data.base import BaseDataset


class COUNTERFACT_MAXDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        
        prompt = row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"])
        equiv_prompt = random.choice(row["paraphrase_prompts"])
        answer = row["requested_rewrite"]["target_new"]["str"]
        unrel_prompt = random.choice(row["neighborhood_prompts"])
        unrel_answer = row["requested_rewrite"]["target_true"]["str"]
        generation_prompts = row["generation_prompts"]
    
        return {
            "edit_tuples": self.tok_tuples(prompt, answer),
            "equiv_tuples": self.tok_tuples(equiv_prompt, answer),
            "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer)
        }
        
    def tok_tuples(
        self,
        prompt: str,
        answer: str,
    ) -> Dict[str, torch.LongTensor]:

        # if isinstance(self.tok, GPT2TokenizerFast):
        answer = " " + answer

        tok_prompt = self.tok(
            prompt,
            return_tensors = "pt",
            truncation=False,
            padding=False
        )
        tok_answer = self.tok(
            answer,
            return_tensors = "pt",
            add_special_tokens = False,
            truncation=False,
            padding=False
        )

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)

        return tok_tuples