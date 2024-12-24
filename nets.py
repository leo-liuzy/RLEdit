from typing import Tuple

import torch
import torch.nn as nn

import numpy as np


# class SelfAttention(nn.Module):
#     def __init__(self, size: int, num_heads: int):
#         super().__init__()
#         self.size = size
#         self.num_heads = num_heads
#         self.scaling = (size // num_heads) ** -0.5  # 缩放因子

#         self.q_proj = nn.Linear(size, size)
#         self.k_proj = nn.Linear(size, size)
#         self.v_proj = nn.Linear(size, size)
#         self.out_proj = nn.Linear(size, size)

#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         """
#         x: 输入张量，形状为 (batch_size, seq_len, size)
#         返回值: 加权后的输出，形状为 (batch_size, seq_len, size)
#         """
#         batch_size, seq_len, size = x.size()

#         # 线性变换得到 Query, Key, Value
#         Q = self.q_proj(x)  # (batch_size, seq_len, size)
#         K = self.k_proj(x)  # (batch_size, seq_len, size)
#         V = self.v_proj(x)  # (batch_size, seq_len, size)

#         # 分成多头
#         Q = Q.view(batch_size, seq_len, self.num_heads, size // self.num_heads).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
#         K = K.view(batch_size, seq_len, self.num_heads, size // self.num_heads).transpose(1, 2)
#         V = V.view(batch_size, seq_len, self.num_heads, size // self.num_heads).transpose(1, 2)

#         # 计算 Attention 分数
#         scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling  # (batch_size, num_heads, seq_len, seq_len)
#         attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

#         # 加权 Value
#         attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

#         # 合并多头
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, size)  # (batch_size, seq_len, size)

#         # 最后一层线性变换
#         output = self.out_proj(attn_output)  # (batch_size, seq_len, size)
#         return output


class RunningMeanStd(nn.Module):

    def __init__(self, size: int):
        super().__init__()

        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("mean", torch.zeros((size)))
        self.register_buffer("var", torch.zeros((size)))
        self.register_buffer("std", torch.zeros((size)))


    def update(self, x: torch.FloatTensor):

        n = self.n + x.shape[0]
        delta = x.mean(0) - self.mean
        self.mean += x.shape[0] * delta / n
        self.var += x.shape[0] * x.var(0) + self.n * x.shape[0] * delta.pow(2) / n
        # self.var += x.shape[0] *  torch.tensor(np.var(x.cpu().numpy(), axis=0, ddof=1)).to('cuda') + self.n * x.shape[0] * delta.pow(2) / n
        self.std = (self.var / (n - 1 + torch.finfo(x.dtype).eps)).sqrt()
        self.n = n


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        return (x - self.mean) / (self.std + torch.finfo(x.dtype).eps)


class MALMENBlock(nn.Module):

    def __init__(self, size: int, rank: int, n_modules: int):
        super().__init__()

        self.A = nn.Parameter(torch.randn(size, rank))
        self.B = nn.Parameter(torch.zeros(rank, size))
        self.bias = nn.Parameter(torch.zeros(size))
        
        self.scale = nn.Embedding(n_modules, size)
        self.shift = nn.Embedding(n_modules, size)
        
        self.scale.weight.data.fill_(1)
        self.shift.weight.data.fill_(0)

        # 引入 Self-Attention 模块
        # self.self_attention = SelfAttention(size, num_heads)


    def forward(
        self,
        y: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> torch.FloatTensor:

        x = y @ self.A @ self.B + self.bias
        x = x.clamp(0)

        # Self-Attention 部分
        # print(x.shape)
        # x = self.self_attention(x)

        x = self.scale(module_idx) * x + self.shift(module_idx)
        # print(f"x:{x.shape}")
        # print(f"y:{y.shape}")
        x = x + y

        return x


class MALMENNet(nn.Module):

    def __init__(
        self,
        key_size: int,
        value_size: int,
        rank: int,
        n_blocks: int,
        n_modules: int,
        lr: float
    ):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        self.normalizer = RunningMeanStd(key_size + value_size)
        self.blocks = nn.ModuleList([
            MALMENBlock(key_size + value_size, rank, n_modules)
            for _ in range(n_blocks)
        ])

        self.lr = nn.Embedding(n_modules, 1)
        self.lamda = nn.Embedding(n_modules, 1)
        
        self.lr.weight.data.fill_(lr)
        self.lamda.weight.data.fill_(0)


    def forward(
        self,
        keys: torch.FloatTensor,
        values_grad: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> Tuple[torch.FloatTensor]:

        # print(f"keys: {keys.shape}")
        # print(f"values_grad: {values_grad.shape}")
        hidden_states = torch.cat((keys, values_grad), -1)
        # print(f"hidden_states: {hidden_states.shape}")
        hidden_states = self.normalizer(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, module_idx)
        return hidden_states.split([self.key_size, self.value_size], -1)