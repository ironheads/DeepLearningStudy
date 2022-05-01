import math
import torch
import torch.nn as nn
import torch.functional as F


'''
    An implementation  of 
'''


class GRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W_zh, self.W_zx, self.b_z = self.get_parameters()
        self.W_rh, self.W_rx, self.b_r = self.get_parameters()
        self.W_hh, self.W_hx, self.b_h = self.get_parameters()
        self.Linear = nn.Linear(hidden_dim, output_dim)  # 全连接层做输出
        self.reset()

    def get_parameters(self, target: str) -> tuple(nn.Parameter, nn.Parameter, nn.Parameter):
        input_dim, hidden_dim, output_dim = self.input_dim, self.hidden_dim, self.output_dim
        return nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim)), nn.Parameter(torch.FloatTensor(input_dim, hidden_dim)), nn.Parameter(torch.FloatTensor(hidden_dim))

    def reset(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        input = input.type(torch.float32)
        Y = []
        h = state
        for x in input:
            z = nn.Sigmoid(h@self.W_zh + x @ self.W_zx + self.b_z)
            r = nn.Sigmoid(h @ self.W_rh + x @ self.W_rx + self.b_r)
            ht = nn.Tanh((h * r) @ self.W_hh + x @ self.W_hx + self.b_h)
            h = (1 - z) * h + z * ht
            y = self.Linear(h)
            Y.append(y)
        return torch.cat(Y, dim=0), h
