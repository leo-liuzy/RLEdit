from typing import Dict
from omegaconf import DictConfig

import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from rnnnets import RNNNet

from editor.base import BaseEditor
from util import get_module, get_shape


def pad_tensor(tensor, target_length, dim=0, padding_value=0):
    """
    在指定维度上将张量填充到目标长度,并返回填充后的张量和对应的 mask。

    参数:
        tensor (torch.Tensor): 输入张量,形状为(..., length, ...)。
        target_length (int): 目标长度。
        dim (int): 要填充的维度,默认为0。
        padding_value (float): 填充值,默认为0。

    返回:
        padded_tensor (torch.Tensor): 填充后的张量,形状为(..., target_length, ...)。
        mask (torch.Tensor): 填充的 mask,形状为(..., target_length, ...)。
    """
    tensor_length = tensor.size(dim)
    if tensor_length >= target_length:
        return tensor.narrow(dim, 0, target_length)
    else:
        padding = target_length - tensor_length
        pad_shape = list(tensor.shape)
        pad_shape[dim] = padding
        pad_tensor = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        mask = torch.cat([torch.ones(tensor_length, dtype=torch.float32, device=tensor.device),
                          torch.zeros(padding, dtype=torch.float32, device=tensor.device)], dim=0)
        return torch.cat([tensor, pad_tensor], dim=dim)


# def pad_tensor(tensor, target_length, dim=1, padding_value=0):
#     """
#     将张量填充到指定维度的目标长度。
    
#     参数:
#         tensor (torch.Tensor): 输入张量，形状为 (..., length, ...)。
#         target_length (int): 要填充到的目标长度。
#         dim (int): 要填充的维度，默认为 1。
#         padding_value (float): 填充值，默认为 0。
    
#     返回:
#         padded_tensor (torch.Tensor): 填充后的张量。
#         original_length (int): 原始张量在指定维度上的长度。
#     """
#     original_length = tensor.size(dim)  # 获取原始长度
#     if original_length >= target_length:
#         # 如果长度超过目标长度，则裁剪
#         return tensor.narrow(dim, 0, target_length), original_length
#     else:
#         # 如果长度不足，则填充
#         pad_size = target_length - original_length
#         pad_shape = list(tensor.shape)
#         pad_shape[dim] = pad_size
#         pad_tensor = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
#         padded_tensor = torch.cat([tensor, pad_tensor], dim=dim)
#         return padded_tensor, original_length


# def unpad_tensor(tensor, original_length, dim=1):
#     """
#     将张量恢复到原始长度。
    
#     参数:
#         tensor (torch.Tensor): 填充后的张量。
#         original_length (int): 原始长度。
#         dim (int): 要恢复的维度，默认为 1。
    
#     返回:
#         unpadded_tensor (torch.Tensor): 恢复后的张量。
#     """
#     return tensor.narrow(dim, 0, original_length)



class NEWMALMEN(BaseEditor):

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        super().__init__(
            config,
            model
        )

        self.net = nn.ModuleDict({
            str(k): RNNNet(
                *k,
                config.editor.hidden_size,
                config.editor.n_blocks,
                v,
                config.editor.lr,
                config.editor.n_gru,
                config.model.model_vec_size,
            )
            for k, v in self.shape_counter.items()
        }).to(config.editor_device)

        self.hidden = {
            str(k): [
                torch.zeros(
                    config.editor.n_gru,  # 对应 GRU 的层数
                    config.editor.batch_size,
                    config.editor.hidden_size,
                    device=config.editor_device
                )
                for _ in range(config.editor.n_blocks)
            ]
            for k in self.shape_counter.keys()
        }

        self.opt = torch.optim.Adam(
            self.net.parameters(),
            config.editor.meta_lr
        )
        
        if config.editor.load_checkpoint:
            self.net.load_state_dict(torch.load(f"checkpoints/{config.model.name}_{config.editor.name}_{str(config.data.n_edits)}_net.pth"))
            self.opt.load_state_dict(torch.load(f"checkpoints/{config.model.name}_{config.editor.name}_{str(config.data.n_edits)}_opt.pth"))
            print("-----Loaded checkpoints-----")


    def reset_hypernet(self):

        self.net = nn.ModuleDict({
            str(k): RNNNet(
                *k,
                self.config.editor.hidden_size,
                self.config.editor.n_blocks,
                v,
                self.config.editor.lr,
                self.config.editor.n_gru,
                self.config.model.model_vec_size,
            )
            for k, v in self.shape_counter.items()
        }).to(self.config.editor_device)

        self.hidden = {
            str(k): [
                torch.zeros(
                    self.config.editor.n_gru,  # 对应 GRU 的层数
                    self.config.editor.batch_size,
                    self.config.editor.hidden_size,
                    device=self.config.editor_device
                )
                for _ in range(self.config.editor.n_blocks)
            ]
            for k in self.shape_counter.keys()
        }
        
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            self.config.editor.meta_lr
        )


    def predict_param_shifts(self) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.model.edit_modules):

            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            keys = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}/{module_idx}_{idx}_keys.pth")
                for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size))
            ])
            values_grad = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}/{module_idx}_{idx}_values_grad.pth")
                for idx in range(math.ceil(self.config.data.n_edits // self.config.data.batch_size))
            ])
            value_diffs = torch.empty((0, net.value_size), device = self.config.editor_device)
            hidden = self.hidden[str(shape)]
            # print(f"kes.shape[0]: {keys.shape[0]}")
            for start_idx in range(0, keys.shape[0], self.config.editor.batch_size):
                end_idx = start_idx + self.config.editor.batch_size
                keys_once = pad_tensor(keys[start_idx:end_idx], self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor.batch_size, 0)
                with torch.no_grad():
                    (pesudo_keys, pesudo_values_grad), hidden = net(
                        keys_once,
                        values_grad_once,
                        layer_idx,
                        hidden
                    )
                    coeffs = - net.lr(layer_idx) * (keys_once * pesudo_keys).sum(-1).unsqueeze(-1)
                # print(f"value_diffs: {value_diffs.shape}")
                value_diffs = torch.cat((value_diffs, coeffs * pesudo_values_grad))
            self.hidden[str(shape)] = [h.detach() for h in hidden]
            with torch.no_grad():
                mat = keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device = self.config.editor_device)
            value_diffs = value_diffs[:keys.shape[0], :]
            param_shift = torch.linalg.solve(mat, keys.T @ value_diffs)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)
            
        # print("complete predict")
        return param_shifts
        
        
    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor]):
        
        # self.opt.zero_grad()
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            keys = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}/{module_idx}_{idx}_keys.pth")
                for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size))
            ])
            values_grad = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}/{module_idx}_{idx}_values_grad.pth")
                for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size))
            ])
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32).to(self.config.editor_device)
            param_shift = param_shifts[module_name].to(self.config.editor_device)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            with torch.no_grad():
                mat = torch.linalg.solve(keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device = self.config.editor_device), module_grad)
                lamda_grad = - net.lamda(layer_idx).exp() * (mat * param_shift).sum()
            value_diffs_grad = keys @ mat
            (lamda_grad * net.lamda(layer_idx)).backward()
            hidden = self.hidden[str(shape)]
            for start_idx in range(0, keys.shape[0], self.config.editor.batch_size):
                end_idx = start_idx + self.config.editor.batch_size
                keys_once = pad_tensor(keys[start_idx:end_idx], self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor.batch_size, 0)
                (pesudo_keys, pesudo_values_grad), hidden = net(
                    keys_once,
                    values_grad_once,
                    layer_idx,
                    hidden
                )
                coeffs = - net.lr(layer_idx) * (keys_once * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diff = coeffs * pesudo_values_grad
                value_diff = value_diff[:keys.shape[0] - start_idx, :]
                (value_diffs_grad[start_idx:end_idx] * value_diff).sum().backward(retain_graph=True)
            self.hidden[str(shape)] = [h.detach() for h in hidden]
            
        clip_grad_norm_(
            self.net.parameters(),
            self.config.editor.max_grad_norm
        )
        # self.opt.step()
        # self.opt.zero_grad()
        # print("complete update")