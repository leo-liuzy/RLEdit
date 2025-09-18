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

import json
import pickle
import functools

def load_jsonlines(fname: str):
    """Read jsonlines file."""
    with open(fname, "r") as f:
        return [json.loads(line) for line in f]
    
def generate_multi_answers(
    context: str,
    answers: List[str],
    config,
    model,
    tokenizer,
    generation_config,
):
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=True)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]

    inputs = {k: v.to(config.model_device) for k, v in inputs.items()}
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

def add_padding(tokenizer, model):
    import transformers

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM) and not isinstance(model, transformers.Qwen2ForCausalLM):
        #     model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight.mean(0)
        # else:
        model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)

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
    add_padding(tok, editor.model)
    
    # test_type = "test_ood_entity"
    test_type = config.test_type
    save_dir = f"results/{config.dataset.name}_{config.model.name}_{config.editor.name}_{config.dataset.n_edits}_{config.num_seq}_ep{config.editor.n_epochs}"
    os.makedirs(save_dir, exist_ok=True)
    fpath = f"{save_dir}/{test_type}_results.xlsx"
    if test_type == "test_id":
        test_set = data_class(config.dataset, f"{os.path.dirname(__file__)}/data/raw/ctrlRE/test_id.json", tok, config.model_device)
        original_test_set = load_jsonlines(f"{os.path.dirname(__file__)}/data/raw/4Ktrain_data_100percent_frozen/test_text_data_id_entity152_rel31.jsonl")
    elif test_type == "test_ood_both":
        test_set = data_class(config.dataset, f"{os.path.dirname(__file__)}/data/raw/ctrlRE/test_ood_both.json", tok, config.model_device)
        original_test_set = load_jsonlines(f"{os.path.dirname(__file__)}/data/raw/4Ktrain_data_100percent_frozen/test_text_data_ood_entity37_rel7.jsonl")

    elif test_type == "test_ood_relation":
        test_set = data_class(config.dataset, f"{os.path.dirname(__file__)}/data/raw/ctrlRE/test_ood_relation.json", tok, config.model_device)
        original_test_set = load_jsonlines(f"{os.path.dirname(__file__)}/data/raw/4Ktrain_data_100percent_frozen/test_text_data_ood-relation_entity152_rel7.jsonl")
    else:
        assert test_type == "test_ood_entity"
        test_set = data_class(config.dataset, f"{os.path.dirname(__file__)}/data/raw/ctrlRE/test_ood_entity.json", tok, config.model_device)
        original_test_set = load_jsonlines(f"{os.path.dirname(__file__)}/data/raw/4Ktrain_data_100percent_frozen/test_text_data_ood-entity_entity37_rel31.jsonl")
        
    
    test_loader = DataLoader(test_set, config.dataset.n_edits, shuffle=False, collate_fn = test_set.collate_fn,drop_last = False)
    

    n_edit_per_generation = config.num_seq * config.dataset.n_edits
    debug_tuples_list = [b for b in test_loader][:config.num_seq]
    total_batches = len(test_loader)
    # import pdb; pdb.set_trace()
    print(f"Checksum model parameter: {checksum_model(editor.model)}")
    
    all_results = []
    test_inputs = []
    for batch_idx, tuples in enumerate(tqdm(test_loader, desc = "Test", ncols = 100)):
        # Cache the edit tuples
        # import pdb; pdb.set_trace()
        editor = editor_class(config, model)
        add_padding(tok, editor.model)
        editor.cache(tuples["edit_tuples"])
        param_shifts = editor.predict_param_shifts()
        editor.edit_model(param_shifts, False)
        # editor.tuples_list.append(tuples)
        editor.opt.zero_grad()

        test_inputs.extend(original_test_set[batch_idx* config.dataset.n_edits: (batch_idx+1)*config.dataset.n_edits])
        
        # Check if we've reached the generation interval or if this is the last batch
        is_generation_interval = (batch_idx + 1) % config.num_seq == 0
        
        is_last_batch = batch_idx == total_batches - 1
        
        if is_generation_interval or is_last_batch:
            # Perform operations every n_edit_per_generation iterations or on the last batch
            
            # Add any other operations you want to perform here
            # For example: editor.apply_edits(), editor.update_model(), etc.
            for t_i, test_input in enumerate(test_inputs):
                for question_key in ["alias_question", "unalias_question"]:
                    for q_i, question in enumerate(test_input["questions"]):
                        post_result_df = generate_multi_answers(
                            context=question[question_key],
                            answers=str(question["answer"]),
                            config=config,
                            model=editor.model,
                            tokenizer=tok,
                            generation_config=generation_config,
                        )
                        
                        post_result_df.insert(0, "question_key", question_key)
                        post_result_df.insert(0, "stage", "post-edit")
                        post_result_df.insert(
                            0, "edit_input", test_input["text"]
                        )
                        all_results.append(post_result_df)
            # import pdb; pdb.set_trace()
            test_inputs = []
            empty_cache(config.editor.cache_dir, config)
            # editor.reset_model()
            gc.collect()
            torch.cuda.empty_cache()
    df = pd.concat(all_results, axis=0, ignore_index=True)
    df.to_excel(fpath, index=False)
    print(f"Saved results to {fpath}")
    

if __name__ == "__main__":
    main()