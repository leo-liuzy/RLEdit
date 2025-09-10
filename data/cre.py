from typing import Dict
import torch

from data.base import BaseDataset


class CREDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        
        text = row["text"]
        prop_qs = row["questions"]
        prop_as = row["answers"]
        unrel_prompt = row["loc"] + "?"
        unrel_answer = row["loc_ans"]
        
        first_qa = self.tok_tuples(prop_qs[0], prop_as[0])
        second_qa = self.tok_tuples(prop_qs[1], prop_as[1])
        # import pdb; pdb.set_trace()
        # a = self.
        return {
            # "edit_tuples": self.tok_tuples(prompt, answer),
            # "equiv_tuples": self.tok_tuples(prop_qs, prop_as),
            "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer)
        }
        

    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:

        
        if isinstance(answer, str):
            answer = " " + answer
        elif isinstance(answer, list):
            answer = [" " + a for a in answer]
        else:
            raise Exception("Un-handled data type")
        # import pdb; pdb.set_trace()
        tok_prompt = self.tok(
            prompt,
            return_tensors="pt",
        )
        tok_answer = self.tok(
            answer,
            return_tensors="pt",
            add_special_tokens=False
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

    # def tok_tuples(
    #     self,
    #     prompts: List[str],
    #     answers: List[str],
    # ) -> Dict[str, torch.LongTensor]:

    #     answer = " " + answer
    #     tok_prompt = self.tok(
    #         prompt,
    #         return_tensors="pt",
    #     )
    #     tok_answer = self.tok(
    #         answer,
    #         return_tensors="pt",
    #         add_special_tokens=False
    #     )

    #     tok_tuples = {
    #         key: torch.cat((value, tok_answer[key][:, :-1]), -1)
    #         for key, value in tok_prompt.items()
    #     }
        
    #     tok_tuples["labels"] = torch.cat((
    #         torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
    #         tok_answer["input_ids"]
    #     ), -1)

    #     return tok_tuples