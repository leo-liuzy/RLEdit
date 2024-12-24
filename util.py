from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.pytorch_utils import Conv1D

def empty_cache(path: str, config):

    dir_path = f"{config.editor.cache_dir}/{config.model.name}_{config.editor.name}_{config.data.n_edits}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    try:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error while clearing cache: {e}")

def get_module(module: nn.Module, module_name: str) -> nn.Module:
    
    for name in module_name.split("."):
        module = getattr(module, name)
    return module

def get_shape(module: Union[nn.Linear, Conv1D]) -> Tuple[int]:
    
    shape = tuple(module.weight.shape)
    return shape[::-1] if isinstance(module, nn.Linear) else shape
    
def cross_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor
):
    if len(logits.shape) == 2:

        return F.binary_cross_entropy_with_logits(logits, labels)

    if len(logits.shape) == 3:

        ans_indice = torch.where(labels != -100)
        
        logits = logits[ans_indice]
        labels = labels[ans_indice]
        
        return F.cross_entropy(logits, labels)

def log(x: torch.FloatTensor) -> torch.FloatTensor:
    return (x + torch.finfo(x.dtype).eps).log()

def kl_div(
    refer_logits: torch.FloatTensor,
    logits: torch.FloatTensor,
    labels: torch.LongTensor
) -> torch.Tensor:
    
    if len(logits.shape) == 2:

        refer_probs = F.sigmoid(refer_logits)
        probs = F.sigmoid(logits)

        return (refer_probs * (log(refer_probs) - log(probs))) + ((1 - refer_probs) * (log(1 - refer_probs) - log(1 - probs)))
    
    if len(logits.shape) == 3:

        ans_indice = torch.where(labels != -100)
        
        refer_logits = refer_logits[ans_indice]
        logits = logits[ans_indice]
        
        refer_log_probs = refer_logits.log_softmax(-1)
        log_probs = logits.log_softmax(-1)
        
        return F.kl_div(
            log_probs,
            refer_log_probs,
            reduction = "batchmean",
            log_target = True
        )
    
def succ_ratios(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    old_labels: torch.LongTensor=None
) -> List[float]:
    
    if old_labels is None:
    
        if len(logits.shape) == 2:

            return ((logits > 0) == labels).squeeze(-1).to("cpu").numpy().tolist()
        
        if len(logits.shape) == 3:

            n_corr = (logits.argmax(-1) == labels).sum(-1)
            n_tokens = (labels != -100).sum(-1)
            # print(f"n_corr: {n_corr}")
            # print(f"n_tokens: {n_tokens}")

            
            return (n_corr / n_tokens).to("cpu").numpy().tolist()
    
    else:

        if len(logits.shape) == 2:

            if old_labels.shape[1] > labels.shape[1]:
                # 裁剪 old_labels 的第二维到与 labels 的第二维一致
                old_labels = old_labels[:, :labels.shape[1]]

            # 获取 logits 中对应 label 和 old_label 的概率
            label_probs = logits[torch.arange(logits.size(0)), labels]
            old_label_probs = logits[torch.arange(logits.size(0)), old_labels]
            
            # 比较 label 的概率是否大于 old_label 的概率
            success = (label_probs > old_label_probs).to(torch.float32)

        if len(logits.shape) == 3:
            # print(logits.shape)
            # print(labels.shape)
            # print(old_labels.shape)

            # valid_mask = (labels != -100) & (old_labels != -100)
            batch_size, seq_len, _ = logits.shape

            # # 确保 old_labels 和 labels 的形状与 logits 的前两个维度匹配
            # if old_labels.shape[1] != seq_len:
            #     old_labels = old_labels[:, :seq_len]  # 截断到 seq_len 的长度
            # if labels.shape[1] != seq_len:
            #     labels = labels[:, :seq_len]  # 截断到 seq_len 的长度

            if old_labels.shape[1] > labels.shape[1]:
                # 裁剪 old_labels 的第二维到与 labels 的第二维一致
                old_labels = old_labels[:, :labels.shape[1]]

            if labels.shape[1] > old_labels.shape[1]:
                # 裁剪 labels 的第二维到与 old_labels 的第二维一致
                # resize_labels = labels[:, :old_labels.shape[1]]
                move = labels.shape[1] - old_labels.shape[1]
                labels = labels[:, :old_labels.shape[1]]
                seq_len -= move
                # label_probs = logits[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), resize_labels]
                # old_labels = F.pad(old_labels, (0, 1))  # (0, 1) 表示在最后一维左侧填充0列,右侧填充1列

            valid_mask = (labels != -100) & (old_labels != -100)
            # 获取 logits 中 labels 和 old_label 对应的概率
            label_probs = logits[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), labels]
            old_label_probs = logits[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), old_labels]

            # 比较 label 的概率是否大于 old_label 的概率
            success = ((label_probs > old_label_probs) & valid_mask).to(torch.float32)

        # 计算每个样本正确预测的 token 数量
        n_corr = success.sum(-1)

        # 计算每个样本中有效的 token 数量（忽略 -100 的部分）
        n_tokens = (labels != -100).sum(-1)

        # print(f"n_corr: {n_corr}")
        # print(f"n_tokens: {n_tokens}")

        # print(n_corr)
        # print(n_tokens)
        # 返回每个样本的成功率
        return (n_corr / n_tokens).to("cpu").numpy().tolist()


class Tracer:

    def __init__(
        self,
        module: nn.Module,
        cache_mask: torch.LongTensor
    ):
        cache_indices = torch.where(cache_mask)

        def forward_hook(
            module: nn.Module,
            inputs: Tuple[torch.FloatTensor],
            outputs: Tuple[torch.FloatTensor]
        ):
            self.keys = inputs[0][cache_indices].detach()
            
        def backward_hook(
            module: nn.Module,
            inputs_grad: Tuple[torch.FloatTensor],
            outputs_grad: Tuple[torch.FloatTensor]
        ):
            self.values_grad = outputs_grad[0][cache_indices].detach()

        self.handles = [
            module.register_forward_hook(forward_hook),
            module.register_full_backward_hook(backward_hook)
        ]


class TracerDict(dict):
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        tuples: Dict[str, torch.LongTensor]
    ):
        
        if any("encoder" in m for m in config.model.edit_modules) and any("decoder" in m for m in config.model.edit_modules):
            
            for module_name in config.model.edit_modules:
                if "encoder" in module_name:
                    cache_mask = tuples["attention_mask"]
                else:
                    cache_mask = tuples["decoder_attention_mask"]
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)

        else:

            if config.editor.token == "ans":
                cache_mask = tuples["labels"] != -100
            else:
                cache_mask = tuples["attention_mask"]

            for module_name in config.model.edit_modules:
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)
            
    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        for v in self.values():
            for h in v.handles:
                h.remove()