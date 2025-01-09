from typing import Dict, List
from omegaconf import DictConfig

from collections import Counter
import numpy as np
import os

from nets import MALMENNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import islice
from safetensors.torch import save_file

from tqdm import tqdm
import wandb

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file

from glue_eval.glue_eval import GLUEEval
import json

from model import make_model, make_model_retrain
from util import (
    get_module,
    get_shape,
    empty_cache,
    TracerDict,
    cross_entropy,
    kl_div,
    succ_ratios
)


class BaseEditor:

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        
        self.config = config
        self.model = model
        self.ori_model = model
        
        shape_counter = Counter()
        self.name2idx = {}
        for module_name in config.model.edit_modules:
            shape = get_shape(get_module(model, module_name))
            # print(f"shape: {shape}")
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1
        
        self.shape_counter = shape_counter

        # 测试20个20个的持续编辑
        self.tuples_list = []


    def edit_model(
        self,
        param_shifts: Dict[str, torch.FloatTensor],
        is_reverse: bool
    ):
        
        for module_name, param_shift in param_shifts.items():
            module = get_module(self.model, module_name)
            if isinstance(module, nn.Linear):
                param_shift = param_shift.T
            if is_reverse:
                param_shift = - param_shift
            module.weight.data += param_shift.to(module.weight.data.dtype)


    def reset_model(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = make_model(self.config.model).to(self.config.model_device)


    def cache(self, tuples: List[Dict[str, torch.LongTensor]]):

        for idx, t in enumerate(tuples):
            
            if "old_labels" in t:
                old_labels = t.pop("old_labels")

            with TracerDict(
                self.model,
                self.config,
                t
            ) as tr:
                logits = self.model(**t)["logits"]
                cross_entropy(logits, t["labels"]).backward()
        
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                keys = tr[module_name].keys.to(torch.float32).to(self.config.editor_device)
                values_grad = tr[module_name].values_grad.to(torch.float32).to(self.config.editor_device)
                self.net[str(shape)].normalizer.update(torch.cat((keys, values_grad), -1))
                dir_path = f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(keys, f"{dir_path}/{module_idx}_{idx}_keys.pth")
                torch.save(values_grad, f"{dir_path}/{module_idx}_{idx}_values_grad.pth")

            try:
                t["old_labels"] = old_labels
            except:
                pass


###########################
## Several TRAIN methods ##
###########################

    def train(self, loader: DataLoader, save=False):
        """
        Original training method for MEND and MALMEN, which is for one-shot knowledge editing.
        """

        for _, tuples in enumerate(tqdm(loader, desc = "Train", ncols = 100)):

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.model.zero_grad()

            gen_losses = []
            self.edit_model(param_shifts, False)
            for t in tuples["equiv_tuples"]:
                if "old_labels" in t:
                    old_labels = t.pop("old_labels")
                logits = self.model(**t)["logits"]
                try:
                    t["old_labels"] = old_labels
                except:
                    pass
                loss = cross_entropy(logits, t["labels"])
                loss.backward()
                gen_losses += [loss.item()]
            self.edit_model(param_shifts, True)

            loc_losses = []
            for t in tuples["unrel_tuples"]:

                if "old_labels" in t:
                    old_labels = t.pop("old_labels")

                with torch.no_grad():
                    refer_logits = self.model(**t)["logits"]

                self.edit_model(param_shifts, False)
                logits = self.model(**t)["logits"]

                try:
                    t["old_labels"] = old_labels
                except:
                    pass

                loss = kl_div(
                    refer_logits,
                    logits,
                    t["labels"]
                )
                (self.config.editor.loc_coef * loss).backward()
                self.edit_model(param_shifts, True)
                loc_losses += [loss.item()]

            self.update_hypernet(param_shifts, update=True)

            wandb.log({
                "gen_loss": np.mean(gen_losses),
                "loc_loss": np.mean(loc_losses)
            })

            if self.config.data.name == "zsre" or self.config.data.name == "counterfact":
                if _ >= 100:
                    break
        
        if save:
            torch.save(self.net, "/root/autodl-tmp/hypernet_origin_malmen_1e6.pt")


    def train_reedit(self, loader: DataLoader, save=False):
        """
        The training method for ReEdit.
        Model the sequential editing as a Markov Devision Process, and use the Paradigm of Reinforce Learning to solve the question.
        """

        sequence_tuples = []
        tot_loss, tot_loc_loss = 0, 0

        max_steps = self.config.num_seq
        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Train", ncols=100, total=max_steps)):

            sequence_tuples.append(tuples)

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.model.zero_grad()

            gen_losses = []
            self.edit_model(param_shifts, False)
            # for tup in sequence_tuples:
            tot_loss_once = 0
            for t in tuples["equiv_tuples"]:
                if "old_labels" in t:
                    old_labels = t.pop("old_labels")
                logits = self.model(**t)["logits"]
                try:
                    t["old_labels"] = old_labels
                except:
                    pass
                loss = cross_entropy(logits, t["labels"])
                #loss.backward()
                gen_losses.append(loss.item())
                tot_loss_once += loss
            tot_loss += tot_loss_once
                # tot_loss += tot_loss_once
                # gen_losses.append(tot_loss.item())
            tot_loss_once.backward()
            self.edit_model(param_shifts, True)

            loc_losses = []
            tot_loc_loss = 0
            # for tup in sequence_tuples:
            tot_loc_loss_once = 0
            for t in tuples["unrel_tuples"]:

                if "old_labels" in t:
                    old_labels = t.pop("old_labels")
                with torch.no_grad():
                    refer_logits = self.model(**t)["logits"]
                
                self.edit_model(param_shifts, False)
                logits = self.model(**t)["logits"]
                try:
                    t["old_labels"] = old_labels
                except:
                    pass
                loss = kl_div(
                    refer_logits,
                    logits,
                    t["labels"]
                )
                #(self.config.editor.loc_coef * loss).backward()
                tot_loc_loss_once += (self.config.editor.loc_coef * loss)
                self.edit_model(param_shifts, True)
                loc_losses += [loss.item()]
            tot_loc_loss += tot_loc_loss_once
            
            # loc_losses.append(tot_loc_loss.item())
            tot_loc_loss_once.backward()
            self.edit_model(param_shifts, False)
            self.update_hypernet(param_shifts, False)

            wandb.log({
                "gen_loss": np.mean(gen_losses),
                "loc_loss": np.mean(loc_losses)
            })

        self.opt.step()
        self.opt.zero_grad()
        # tot_loss.backward()
        # tot_loc_loss.backward()


    def train_new(self, loader: DataLoader, save=False):
        """
        A new idea but proved to be useless.
        """

        dir_path = f"{self.config.editor.cache_dir_2}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}"

        sequence_tuples = []
        tot_loss, tot_loc_loss = 0, 0

        max_steps = self.config.num_seq
        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Train", ncols=100, total=max_steps)):

            sequence_tuples.append(tuples)

            # 从这里加入过去的知识一起进行cache，产生序列化微调梯度
            # 每次cache时都使用最初始的LM计算梯度，进行超网络预测，得到迄今为止积累的参数变化量并保存
            # 真正的本轮参数变化量是积累参数变化量减去保存的上次参数变化量

            tuples_now = tuples["edit_tuples"]

            try:
                tuples_old = torch.load(f"{dir_path}/knowledge_tuple_{_-1}.pt")
                os.remove(f"{dir_path}/knowledge_tuple_{_-1}.pt")
                merged_tuples = tuples_now + tuples_old
                # save_file(self.model.state_dict(), "/NAS/ruip/modified_model/model.safetensors") # 保存现在的模型
                # self.reset_model() # 读取初始模型
                self.cache(merged_tuples) # 使用初始模型获得微调梯度
                param_shifts_now = self.predict_param_shifts() # 使用初始模型预测参数变化
                # config = AutoConfig.from_pretrained("/NAS/ruip/modified_model")
                # state_dict = load_file("/NAS/ruip/modified_model/model.safetensors")
                # del self.model # 清除初始模型
                # torch.cuda.empty_cache()
                # self.model = AutoModelForCausalLM.from_pretrained(
                #     pretrained_model_name_or_path=None, 
                #     config=config,
                #     state_dict=state_dict
                # ) # 读取现在的模型
                # del state_dict
                # torch.cuda.empty_cache()
                # self.model.to("cuda")
            except:
                merged_tuples = tuples_now
                self.cache(merged_tuples) # 使用初始模型获得微调梯度
                param_shifts_now = self.predict_param_shifts() # 使用初始模型预测参数变化

            tuples_old = merged_tuples
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(tuples_old, f"{dir_path}/knowledge_tuple_{_}.pt")
            del tuples_old, merged_tuples, tuples_now

            try:
                param_shifts_past = torch.load(f"{dir_path}/param_shifts_{_-1}.pt")
                os.remove(f"{dir_path}/param_shifts_{_-1}.pt")
                # param_shifts = {key: param_shifts_now[key] - param_shifts_past[key] for key in param_shifts_now.keys()}
                param_shifts = param_shifts_now
            except:
                param_shifts = param_shifts_now

            param_shifts_past = param_shifts_now
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(param_shifts_past, f"{dir_path}/param_shifts_{_}.pt")
            del param_shifts_past, param_shifts_now

            self.model.zero_grad()

            gen_losses = []
            self.edit_model(param_shifts, False)
            tot_loss_once = 0
            for t in tuples["equiv_tuples"]:
                if "old_labels" in t:
                    old_labels = t.pop("old_labels")
                logits = self.model(**t)["logits"]
                try:
                    t["old_labels"] = old_labels
                except:
                    pass
                loss = cross_entropy(logits, t["labels"])
                #loss.backward()
                gen_losses.append(loss.item())
                tot_loss_once += loss
            tot_loss += tot_loss_once
                # tot_loss += tot_loss_once
                # gen_losses.append(tot_loss.item())
            tot_loss_once.backward()
            self.edit_model(param_shifts, True)

            loc_losses = []
            tot_loc_loss = 0
            # for tup in sequence_tuples:
            tot_loc_loss_once = 0
            for t in tuples["unrel_tuples"]:

                if "old_labels" in t:
                    old_labels = t.pop("old_labels")
                with torch.no_grad():
                    refer_logits = self.model(**t)["logits"]
                
                self.edit_model(param_shifts, False)
                logits = self.model(**t)["logits"]
                try:
                    t["old_labels"] = old_labels
                except:
                    pass
                loss = kl_div(
                    refer_logits,
                    logits,
                    t["labels"]
                )
                #(self.config.editor.loc_coef * loss).backward()
                tot_loc_loss_once += (self.config.editor.loc_coef * loss)
                self.edit_model(param_shifts, True)
                loc_losses += [loss.item()]
            tot_loc_loss += tot_loc_loss_once
            
            # loc_losses.append(tot_loc_loss.item())
            tot_loc_loss_once.backward()
            self.edit_model(param_shifts, False)
            self.update_hypernet(param_shifts, False)

            wandb.log({
                "gen_loss": np.mean(gen_losses),
                "loc_loss": np.mean(loc_losses)
            })

        self.opt.step()
        self.opt.zero_grad()
        # tot_loss.backward()
        # tot_loc_loss.backward()


###########################
## Several VALID methods ##
###########################

    def valid(self, loader: DataLoader):
        """
        The original valid method for MEND and MALMEN, which just valid the editing of single knowledge.
        """
          
        for tuples in tqdm(loader, desc = "Valid", ncols = 100):

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.edit_model(param_shifts, False)
            edit_succs, gen_succs, loc_succs = [], [], []
            for k, s in zip(
                ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                [edit_succs, gen_succs, loc_succs]
            ):
                for t in tuples[k]:
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    s += succ_ratios(logits, t["labels"])
                    
            self.edit_model(param_shifts, True)
            
            wandb.log({
                "ES": np.mean(edit_succs),
                "GS": np.mean(gen_succs),
                "LS": np.mean(loc_succs)
            })


    def sequential_valid_full(self, loader: DataLoader):
        """
        Valid the entire knowledge sequence, with the full curve showed.
        """

        max_steps = self.config.num_seq
        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

            if self.config.glue_step > 0:
                if _ == 0 or (_+1) % self.config.glue_step == 0:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                    glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                    out_file = f"/data/ruip/NewMend/glue_eval/results/{self.config.model.name}_{_}_{self.config.data.name}"
                    if not os.path.exists(out_file):
                        os.makedirs(out_file, exist_ok=True)
                    out_file = f"/data/ruip/NewMend/glue_eval/results/{self.config.model.name}_{_}_{self.config.data.name}/glue.json"
                    glue_results = {'edit_num': -1}
                    glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    with open(out_file, "w") as f:
                        json.dump(glue_results, f, indent=4)

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.edit_model(param_shifts, False)
            self.tuples_list.append(tuples)
            # tuples_list.append(tuples)
            edit_succs, gen_succs, loc_succs = [], [], []
            for k, s in zip(
                ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                [edit_succs, gen_succs, loc_succs]
            ):
                for tuple in self.tuples_list:
                    for t in tuple[k]:
                        # print(t["labels"].shape)
                        # print(t["old_labels"].shape)
                        if "old_labels" in t:
                            old_labels = t.pop("old_labels")
                        with torch.no_grad():
                            logits = self.model(**t)["logits"]
                        try:
                            t["old_labels"] = old_labels
                        except:
                            pass
                        if self.config.data.name == "counterfact":
                            t["old_labels"] = old_labels
                            # print(f"logits.shape = {logits.shape}")
                            s += succ_ratios(logits, t["labels"], t["old_labels"])
                        else:
                            s += succ_ratios(logits, t["labels"])

            wandb.log({
                "ES": np.mean(edit_succs),
                "GS": np.mean(gen_succs),
                "LS": np.mean(loc_succs)
            })

            self.opt.zero_grad()


    def sequential_valid(self, loader: DataLoader):
        """
        Valid the entire knowledge sequence, with just final results showed.
        """

        max_steps = self.config.num_seq
        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

            if self.config.glue_step > 0:
                if _ == 0 or (_+1) % self.config.glue_step == 0:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                    glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                    out_file = f"/data/ruip/NewMend/glue_eval/results/{self.config.model.name}_{_}_{self.config.data.name}"
                    if not os.path.exists(out_file):
                        os.makedirs(out_file, exist_ok=True)
                    out_file = f"/data/ruip/NewMend/glue_eval/results/{self.config.model.name}_{_}_{self.config.data.name}/glue.json"
                    glue_results = {'edit_num': -1}
                    glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    with open(out_file, "w") as f:
                        json.dump(glue_results, f, indent=4)

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.edit_model(param_shifts, False)
            self.tuples_list.append(tuples)
            self.opt.zero_grad()

            if _ == 20 or _ == 30 or _ == 50 or _ == 100:
                edit_succs, gen_succs, loc_succs = [], [], []
                for k, s in zip(
                    ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                    [edit_succs, gen_succs, loc_succs]
                ):
                    for tuple in self.tuples_list:
                        for t in tuple[k]:
                            if "old_labels" in t:
                                old_labels = t.pop("old_labels")
                            with torch.no_grad():
                                logits = self.model(**t)["logits"]
                            try:
                                t["old_labels"] = old_labels
                            except:
                                pass
                            if self.config.data.name == "counterfact":
                                t["old_labels"] = old_labels
                                # print(f"logits.shape = {logits.shape}")
                                s += succ_ratios(logits, t["labels"], t["old_labels"])
                            else:
                                s += succ_ratios(logits, t["labels"])

                wandb.log({
                    "ES": np.mean(edit_succs),
                    "GS": np.mean(gen_succs),
                    "LS": np.mean(loc_succs)
                })

        edit_succs, gen_succs, loc_succs = [], [], []
        for k, s in zip(
            ["edit_tuples", "equiv_tuples", "unrel_tuples"],
            [edit_succs, gen_succs, loc_succs]
        ):
            for tuple in self.tuples_list:
                for t in tuple[k]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    if self.config.data.name == "counterfact":
                        t["old_labels"] = old_labels
                        # print(f"logits.shape = {logits.shape}")
                        s += succ_ratios(logits, t["labels"], t["old_labels"])
                    else:
                        s += succ_ratios(logits, t["labels"])

        wandb.log({
            "ES": np.mean(edit_succs),
            "GS": np.mean(gen_succs),
            "LS": np.mean(loc_succs)
        })


#########################
## Several RUN methods ##
#########################

    def run_single(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Original run function in MEND and MALMEN, single edit, which means training a hypernet and edit 1 batch of knowledge.
        """

        empty_cache(self.config.editor.cache_dir, self.config)
        self.train(train_loader)
        for _ in range(self.config.editor.n_epochs):
            self.valid(valid_loader)

        torch.save(self.net.state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.data.n_edits)}_net.pth")
        torch.save(self.opt.state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.data.n_edits)}_opt.pth")


    def run_sequential(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use MEND or MALMEN to complete sequential editing task.
        Just train the hypernet on the original LLM, then freeze it.
        """

        empty_cache(self.config.editor.cache_dir, self.config)
        for _ in range(self.config.editor.n_epochs):
            self.train(train_loader)

            self.reset_model()

            if self.config.editor.full_curve == True:
                self.sequential_valid_full(valid_loader)
            else:
                self.sequential_valid(valid_loader)

            empty_cache(self.config.editor.cache_dir, self.config)
            self.reset_hypernet()

    
    def run_sequential_retrain_full(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use MEND or MALMEN to complete sequential editing task.
        Before editing the next batch of knowledge, we retrain the hypernet on post-edited LLM in order to keep its ability.
        This function will show the full curve.
        """

        empty_cache(self.config.editor.cache_dir, self.config)
        for _ in range(self.config.editor.n_epochs):
            
            max_steps = self.config.num_seq
            limited_loader = islice(valid_loader, max_steps)

            for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

                if _ % 3 == 0:
                    self.train(train_loader)

                if self.config.glue_step > 0:
                    # if _ == 0 or (_+1) % self.config.glue_step == 0:
                    if (_+1) % self.config.glue_step == 0:
                        tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                        glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                        out_file = f"/data/ruip/NewMend/glue_eval/results/{self.config.model.name}_{_}_{self.config.data.name}"
                        if not os.path.exists(out_file):
                            os.makedirs(out_file, exist_ok=True)
                        out_file = f"/data/ruip/NewMend/glue_eval/results/{self.config.model.name}_{_}_{self.config.data.name}/glue.json"
                        glue_results = {'edit_num': -1}
                        glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                        with open(out_file, "w") as f:
                            json.dump(glue_results, f, indent=4)

                self.cache(tuples["edit_tuples"])
                param_shifts = self.predict_param_shifts()
                self.edit_model(param_shifts, False)
                self.tuples_list.append(tuples)

                if _ == 19 or _ == 30 or _ == 50 or _ == 100:
                    edit_succs, gen_succs, loc_succs = [], [], []
                    for k, s in zip(
                        ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                        [edit_succs, gen_succs, loc_succs]
                    ):
                        for tuple in self.tuples_list:
                            for t in tuple[k]:
                                if "old_labels" in t:
                                    old_labels = t.pop("old_labels")
                                with torch.no_grad():
                                    logits = self.model(**t)["logits"]
                                try:
                                    t["old_labels"] = old_labels
                                except:
                                    pass
                                if self.config.data.name == "counterfact":
                                    t["old_labels"] = old_labels
                                    # print(f"logits.shape = {logits.shape}")
                                    s += succ_ratios(logits, t["labels"], t["old_labels"])
                                else:
                                    s += succ_ratios(logits, t["labels"])

                    wandb.log({
                        "ES": np.mean(edit_succs),
                        "GS": np.mean(gen_succs),
                        "LS": np.mean(loc_succs)
                    })

                # edit_succs, gen_succs, loc_succs = [], [], []
                # for k, s in zip(
                #     ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                #     [edit_succs, gen_succs, loc_succs]
                # ):
                #     for tuple in self.tuples_list:
                #         for t in tuple[k]:
                #             # print(t["labels"].shape)
                #             # print(t["old_labels"].shape)
                #             if "old_labels" in t:
                #                 old_labels = t.pop("old_labels")
                #             with torch.no_grad():
                #                 logits = self.model(**t)["logits"]
                #             try:
                #                 t["old_labels"] = old_labels
                #             except:
                #                 pass
                #             if self.config.data.name == "counterfact":
                #                 t["old_labels"] = old_labels
                #                 # print(f"logits.shape = {logits.shape}")
                #                 s += succ_ratios(logits, t["labels"], t["old_labels"])
                #             else:
                #                 s += succ_ratios(logits, t["labels"])

                # wandb.log({
                #     "ES": np.mean(edit_succs),
                #     "GS": np.mean(gen_succs),
                #     "LS": np.mean(loc_succs)
                # })
                if _ % 3 == 0:
                    empty_cache(self.config.editor.cache_dir, self.config)
                    self.opt.zero_grad()
                    self.reset_hypernet()

    
    def run_sequential_retrain(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use MEND or MALMEN to complete sequential editing task.
        Before editing the next batch of knowledge, we retrain the hypernet on post-edited LLM in order to keep its ability.
        This function will only show final results.
        """

        max_steps = self.config.num_seq
        limited_loader = islice(valid_loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

            self.train(train_loader)

            if self.config.glue_step > 0:
                if _ == 0 or (_+1) % self.config.glue_step == 0:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                    glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                    out_file = f"/data/ruip/NewMend/glue_eval/results/{self.config.model.name}_{_}_{self.config.data.name}"
                    if not os.path.exists(out_file):
                        os.makedirs(out_file, exist_ok=True)
                    out_file = f"/data/ruip/NewMend/glue_eval/results/{self.config.model.name}_{_}_{self.config.data.name}/glue.json"
                    glue_results = {'edit_num': -1}
                    glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    with open(out_file, "w") as f:
                        json.dump(glue_results, f, indent=4)

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.edit_model(param_shifts, False)
            self.tuples_list.append(tuples)
            self.opt.zero_grad()
            self.reset_hypernet()

        edit_succs, gen_succs, loc_succs = [], [], []
        for k, s in zip(
            ["edit_tuples", "equiv_tuples", "unrel_tuples"],
            [edit_succs, gen_succs, loc_succs]
        ):
            for tuple in self.tuples_list:
                for t in tuple[k]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    if self.config.data.name == "counterfact":
                        t["old_labels"] = old_labels
                        # print(f"logits.shape = {logits.shape}")
                        s += succ_ratios(logits, t["labels"], t["old_labels"])
                    else:
                        s += succ_ratios(logits, t["labels"])

        wandb.log({
            "ES": np.mean(edit_succs),
            "GS": np.mean(gen_succs),
            "LS": np.mean(loc_succs)
        })

    def run_reedit(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use ReEdit to complete sequential editing task.
        """
        
        for _ in tqdm(range(self.config.editor.n_epochs), desc = "epoch"):

            self.reset_model()
            self.train_reedit(train_loader)

            self.reset_model()
            if self.config.editor.save_checkpoint:
                torch.save(self.net.state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.data.n_edits)}_net.pth")
                torch.save(self.opt.state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.data.n_edits)}_opt.pth")
                print("-----Saved checkpoints-----")
                
            if self.config.editor.full_curve == True:
                self.sequential_valid_full(valid_loader)
            else:
                self.sequential_valid(valid_loader)

            empty_cache(self.config.editor.cache_dir, self.config)
        self.reset_hypernet()


    def run_test(self, valid_loader: DataLoader):
        """
        Just include valid process, which needs you saving the hypernet locally.
        """
        
        for _ in tqdm(range(self.config.editor.n_test), desc = "test_num"):
            empty_cache(self.config.editor.cache_dir, self.config)
            self.reset_model()

            if self.config.editor.full_curve == True:
                self.sequential_valid_full(valid_loader)
            else:
                self.sequential_valid(valid_loader)

        self.reset_hypernet()


    def run_preedited(self, train_loader: DataLoader, valid_loader: DataLoader):
        
        max_steps = self.config.num_seq
        limited_loader = islice(valid_loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

            self.tuples_list.append(tuples)
            self.opt.zero_grad()

            if _ == 20 or _ == 30 or _ == 50 or _ == 100:
                edit_succs, gen_succs, loc_succs = [], [], []
                for k, s in zip(
                    ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                    [edit_succs, gen_succs, loc_succs]
                ):
                    for tuple in self.tuples_list:
                        for t in tuple[k]:
                            if "old_labels" in t:
                                old_labels = t.pop("old_labels")
                            with torch.no_grad():
                                logits = self.model(**t)["logits"]
                            try:
                                t["old_labels"] = old_labels
                            except:
                                pass
                            if self.config.data.name == "counterfact":
                                t["old_labels"] = old_labels
                                # print(f"logits.shape = {logits.shape}")
                                s += succ_ratios(logits, t["labels"], t["old_labels"])
                            else:
                                s += succ_ratios(logits, t["labels"])

                wandb.log({
                    "ES": np.mean(edit_succs),
                    "GS": np.mean(gen_succs),
                    "LS": np.mean(loc_succs)
                })

        edit_succs, gen_succs, loc_succs = [], [], []
        for k, s in zip(
            ["edit_tuples", "equiv_tuples", "unrel_tuples"],
            [edit_succs, gen_succs, loc_succs]
        ):
            for tuple in self.tuples_list:
                for t in tuple[k]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    if self.config.data.name == "counterfact":
                        t["old_labels"] = old_labels
                        # print(f"logits.shape = {logits.shape}")
                        s += succ_ratios(logits, t["labels"], t["old_labels"])
                    else:
                        s += succ_ratios(logits, t["labels"])

        wandb.log({
            "ES": np.mean(edit_succs),
            "GS": np.mean(gen_succs),
            "LS": np.mean(loc_succs)
        })