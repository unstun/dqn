from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int = 128, hidden_layers: int = 2):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int = 128, hidden_layers: int = 2):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.feature = nn.Sequential(*layers)

        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))
