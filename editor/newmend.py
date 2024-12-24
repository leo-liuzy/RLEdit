from typing import Dict

import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from editor.base import BaseEditor
from util import get_module, get_shape


# def pad_tensor(tensor, target_length, dim=0, padding_value=0):
#     """
#     在指定维度上将张量填充到目标长度,并返回填充后的张量和对应的 mask。

#     参数:
#         tensor (torch.Tensor): 输入张量,形状为(..., length, ...)。
#         target_length (int): 目标长度。
#         dim (int): 要填充的维度,默认为0。
#         padding_value (float): 填充值,默认为0。

#     返回:
#         padded_tensor (torch.Tensor): 填充后的张量,形状为(..., target_length, ...)。
#         mask (torch.Tensor): 填充的 mask,形状为(..., target_length, ...)。
#     """
#     tensor_length = tensor.size(dim)
#     if tensor_length >= target_length:
#         return tensor.narrow(dim, 0, target_length)
#     else:
#         padding = target_length - tensor_length
#         pad_shape = list(tensor.shape)
#         pad_shape[dim] = padding
#         pad_tensor = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
#         mask = torch.cat([torch.ones(tensor_length, dtype=torch.float32, device=tensor.device),
#                           torch.zeros(padding, dtype=torch.float32, device=tensor.device)], dim=0)
#         return torch.cat([tensor, pad_tensor], dim=dim)


class NEWMEND(BaseEditor):

    def predict_param_shifts(self) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            hidden = self.hidden[str(shape)]
            param_shift = torch.zeros((net.key_size, net.value_size), device = self.config.editor_device)
            for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size)):
                keys = torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}/{module_idx}_{idx}_keys.pth")
                values_grad = torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}/{module_idx}_{idx}_values_grad.pth")
                keys_once = pad_tensor(keys, self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad, self.config.editor.batch_size, 0)
                with torch.no_grad():
                    (pesudo_keys, pesudo_values_grad), hidden = net(
                        keys_once,
                        values_grad_once,
                        layer_idx,
                        hidden
                    )
                    # print(f"pesudo_keys: {pesudo_keys.shape}")
                    # pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
                    param_shift += - net.lr(layer_idx) * pesudo_keys.T @ pesudo_values_grad
            self.hidden[str(shape)] = [h.detach() for h in hidden]
            param_shifts[module_name] = param_shift

        return param_shifts
    

    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor]):
        
        self.opt.zero_grad()
        for module_idx, module_name in enumerate(self.config.model.edit_modules,):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            for idx in range(math.ceil(self.config.data.n_edits / self.config.data.batch_size)):
                keys = torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}/{module_idx}_{idx}_keys.pth")
                values_grad = torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.data.n_edits}/{module_idx}_{idx}_values_grad.pth")
                pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
                param_shift = - net.lr(layer_idx) * pesudo_keys.T @ pesudo_values_grad
                (module_grad * param_shift).sum().backward()

        clip_grad_norm_(
            self.net.parameters(),
            self.config.editor.max_grad_norm
        )
        self.opt.step()