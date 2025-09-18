import os
import hydra
from omegaconf import DictConfig, OmegaConf
import importlib
from data.base import make_loader
from model import make_model
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import torch

from util import (
    get_module,
    get_shape,
    empty_cache,
    cross_entropy,
    kl_div,
    succ_ratios
)
import hashlib

from typing import List
import pandas as pd

def checksum_model(model):
    return sum([p.sum() for p in model.parameters()]).item()

def generate_multi_answers(
    context: str,
    answers: List[str],
    config,
    model,
    tokenizer,
    generation_config,
):
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.gen_w_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]

    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    print(
        "Input for generation:",
        "[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"])) + "]",
    )
    print("Label for generation:", "[" + str(answers) + "]")
    print("--------------------")

    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    generated_texts = [t.replace(ctx_decoded, "") for t in generated_texts]
    predicted_answer = generated_texts[0]
    if hasattr(config, "add_icl") and config.add_icl:
        # if using ICL, extract by the first new line
        if "\n" in predicted_answer:
            predicted_answer = predicted_answer[: predicted_answer.find("\n")]

    model_response = pd.DataFrame(
        [
            {
                "question": context,
                "answer": answers,
                "predicted_answer_idx": 0,
                "predicted_answer": predicted_answer.strip(),
            }
        ]
    )
    return model_response  # score_df(model_response)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    
    data_module = importlib.import_module(f"data.{config.dataset.name}")
    data_class = getattr(data_module, f"{config.dataset.name.upper()}Dataset")

    # train_loader, valid_loader = make_loader(config, data_class)
    # import pdb; pdb.set_trace()
    model = make_model(config.model).to(config.model_device)

    editor_module = importlib.import_module(f"editor.{config.editor.name}")
    editor_class = getattr(editor_module, config.editor.name.upper())
    editor = editor_class(config, model)

    # editor.run(train_loader, valid_loader)
    tok = AutoTokenizer.from_pretrained(config.model.name_or_path)
    generation_config = GenerationConfig(
        do_sample=False,  # Greedy
        top_k=None,
        top_p=None,
        temperature=None,
        max_new_tokens=20,
        num_return_sequences=1,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    
    test_set = data_class(config.dataset, "/u/zliu/datastor1/RLEdit/data/raw/ctrlRE/test_id.json", tok, config.model_device)
    test_loader = DataLoader(test_set, config.dataset.n_edits, shuffle=False, collate_fn = test_set.collate_fn,drop_last = False)
    
    n_edit_per_generation = config.num_seq * config.dataset.n_edits
    debug_tuples_list = [b for b in test_loader][:config.num_seq]
    total_batches = len(test_loader)
    # import pdb; pdb.set_trace()
    print(f"Checksum model parameter: {checksum_model(editor.model)}")
    edit_succs, gen_succs, loc_succs = [], [], []
    for k, s in zip(
        ["edit_tuples", "equiv_tuples", "unrel_tuples"],
        [edit_succs, gen_succs, loc_succs]
    ):
        for tuple in debug_tuples_list:
            for t in tuple[k]:
                if "old_labels" in t:
                    old_labels = t.pop("old_labels")
                with torch.no_grad():
                    logits = editor.model(**t)["logits"]
                try:
                    t["old_labels"] = old_labels
                except:
                    pass
                if config.dataset.name == "counterfact":
                    t["old_labels"] = old_labels
                    s += succ_ratios(logits, t["labels"], t["old_labels"])
                else:
                    s += succ_ratios(logits, t["labels"])
    import pdb; pdb.set_trace()
    for batch_idx, tuples in enumerate(tqdm(test_loader, desc = "Test", ncols = 100)):
        # Cache the edit tuples
        
        editor.cache(tuples["edit_tuples"])
        param_shifts = editor.predict_param_shifts()
        editor.edit_model(param_shifts, False)
        editor.tuples_list.append(tuples)
        editor.opt.zero_grad()
        
        # Check if we've reached the generation interval or if this is the last batch
        is_generation_interval = (batch_idx + 1) % config.num_seq == 0
        is_last_batch = batch_idx == total_batches - 1
        
        if is_generation_interval or is_last_batch:
            # Perform operations every n_edit_per_generation iterations or on the last batch
            
            # Add any other operations you want to perform here
            # For example: editor.apply_edits(), editor.update_model(), etc.
            import pdb; pdb.set_trace()
            
            empty_cache(config.editor.cache_dir, config)
            editor.reset_model()
            gc.collect()
            torch.cuda.empty_cache()
    
    

if __name__ == "__main__":
    main()