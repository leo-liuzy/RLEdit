from typing import Union, Tuple, List, Dict
import torch
import math
from data.base import BaseDataset


class CREDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        
        text = row["text"]
        prop_qs = row["questions"]
        prop_as = row["answers"]
        unrel_prompt = row["loc"] + "?"
        unrel_answer = row["loc_ans"]
        
        
        prop_qas_processed = [
            self.tok_tuples(prop_q, prop_a)
            for prop_q, prop_a in zip(prop_qs, prop_as)
        ]
        # prop_qas_collated = self.collate_fn(prop_qas_processed)["equiv_tuples"][0]
        # import pdb; pdb.set_trace()
        return {
            "edit_tuples": self.tok_text(text),
            "equiv_tuples": prop_qas_processed,
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
            import pdb; pdb.set_trace()
            raise Exception("Un-handled data type")
        
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
    
    def tok_text(
        self,
        text: str,
    ) -> Dict[str, torch.LongTensor]:

        tok_prompt = self.tok(
            text,
            return_tensors="pt",
        )
        
        # shift left
        tok_prompt["labels"] = tok_prompt["input_ids"][:, 1:]
        tok_prompt["input_ids"] = tok_prompt["input_ids"][:, :-1]
        tok_prompt["attention_mask"] = tok_prompt["attention_mask"][:, 1:]

        return tok_prompt


    def collate_fn(
        self,
        tuples: Tuple[Dict[str, Dict[str, torch.LongTensor]]]
    ) -> Dict[str, List[Dict[str, torch.LongTensor]]]:
        
        # import pdb; pdb.set_trace()
        collated_tuples: Dict[str, List[Dict[str, torch.LongTensor]]] = {
            k: sorted(
                [t[k] for t in tuples],
                key = lambda x: x["attention_mask"].sum().item(),
                reverse = True
            )
            for k in ["edit_tuples", "unrel_tuples"]
        }
        collated_tuples["equiv_tuples"] = [qa for t in tuples for qa in t["equiv_tuples"]]
        assert self.config.n_edits == self.config.batch_size, "Unhandled scenario: n_edits != batch_size"
        
        ret = {
            k: [
                self.pad_tok_tuples(v[:])
            ]
            for k, v in collated_tuples.items()
        }
        # import pdb; pdb.set_trace()
        return ret