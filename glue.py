import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'
from transformers import AutoModelForCausalLM, AutoTokenizer
from glue_eval.glue_eval import GLUEEval
# import torch
import json


def main():
    model = AutoModelForCausalLM.from_pretrained("/data/ruip/Gemma-2-9B").to("cuda")
    tok = AutoTokenizer.from_pretrained("/data/ruip/Gemma-2-9B")
    out_file = f"/data/ruip/NewMend/glue_eval/results/gemma/glue.json"
    glue_eval = GLUEEval(model, tok, number_of_tests = 100)
    glue_results = {'edit_num': -1}
    glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
    with open(out_file, "w") as f:
        json.dump(glue_results, f, indent=4)

if __name__=="__main__":
    main()