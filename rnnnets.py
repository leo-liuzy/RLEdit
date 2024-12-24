from typing import Tuple

import torch
import torch.nn as nn


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
        self.std = (self.var / (n - 1 + torch.finfo(x.dtype).eps)).sqrt()
        self.n = n


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        return (x - self.mean) / (self.std + torch.finfo(x.dtype).eps)


class MALMENBlock(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, n_modules: int, n_blocks: int, model_vec_size: int):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers=n_blocks, batch_first=True, dropout=0.4)
        
        self.scale = nn.Embedding(n_modules, hidden_size)
        self.shift = nn.Embedding(n_modules, hidden_size)
        self.output_layer = nn.Linear(hidden_size, model_vec_size)
        
        self.scale.weight.data.fill_(1)
        self.shift.weight.data.fill_(0)


    def forward(
        self,
        y: torch.FloatTensor,
        module_idx: torch.LongTensor,
        hidden: torch.FloatTensor = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        
        # y_pad, y_mask = pad_tensor(y, max_length, dim=0)
        #print(f"y.unsqueeze: {y.unsqueeze(1).shape}")
        # try:
        #     #print(f"hidden: {hidden.shape}")
        # except:
        #     pass
        x, hidden = self.gru(y.unsqueeze(1), hidden)
        # print(f"x:{x.shape}")
        # print(f"y:{y.shape}")
        # y, y_mask = pad_tensor(y, max_length, dim=0)
        x = x.squeeze(1)
        x = self.scale(module_idx) * x + self.shift(module_idx)
        x = self.output_layer(x)  # 新增的线性层
        # x, x_mask = pad_tensor(x, max_length, dim=0)
        # print(f"padded_x:{x.shape}")
        # print(f"padded_y:{y.shape}")
        # # x = x * x_mask.unsqueeze(-1)
        # # y = y * y_mask.unsqueeze(-1)
        # print(f"x: {x.shape}")
        # print(f"y: {y.shape}")
        x = x + y

        return x, hidden


class RNNNet(nn.Module):

    def __init__(
        self,
        key_size: int,
        value_size: int,
        hidden_size: int,
        n_blocks: int,
        n_modules: int,
        lr: float,
        n_gru: int,
        model_vec_size: int,
    ):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        self.normalizer = RunningMeanStd(key_size + value_size)
        # self.normalizers = nn.ModuleList([
        #     RunningMeanStd(key_size + value_size) for _ in range(n_blocks)
        # ])

        self.blocks = nn.ModuleList([
            MALMENBlock(key_size + value_size, hidden_size, n_modules, n_gru, model_vec_size)
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
        module_idx: torch.LongTensor,
        hidden: torch.FloatTensor = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # print(f"keys: {keys.shape}")
        # print(f"values_grad: {values_grad.shape}")
        hidden_states = torch.cat((keys, values_grad), -1)
        # print(f"hidden_states: {hidden_states.shape}")
        hidden_states = self.normalizer(hidden_states)

        if hidden is None:
            hidden = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            # hidden_states = self.normalizers[i](hidden_states)
            hidden_states, hidden[i] = block(hidden_states, module_idx, hidden[i])

        return hidden_states.split([self.key_size, self.value_size], -1), hidden