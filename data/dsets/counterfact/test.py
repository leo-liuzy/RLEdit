import json
import random


path = "../fever/fever_eval.json"
# path = "counterfact.json"
# path = "../zsre/zsre_eval.json"
with open(path) as file:
    data = json.load(file)
row = data[0]
        
# prompt = row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"])
# equiv_prompt = random.choice(row["paraphrase_prompts"])
# answer = row["requested_rewrite"]["target_new"]["str"]
# unrel_prompt = random.choice(row["neighborhood_prompts"])
# unrel_answer = row["requested_rewrite"]["target_true"]["str"]

# prompt = row["src"]
# equiv_prompt = row["rephrase"]
# answer = row["ans"]
# unrel_prompt = row["loc"] + "?"
# unrel_answer = row["loc_ans"]

prompt = row["prompt"]
equiv_prompt = random.choice(row["equiv_prompt"])
unrel_prompt = row["unrel_prompt"]
alt = row["alt"]
ans = row["unrel_ans"]

print(prompt)
print(equiv_prompt)
print(alt)
print(unrel_prompt)
print(ans)