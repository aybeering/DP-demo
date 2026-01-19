import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """输出能量标量 (batch_size, 1)。"""
        return self.net(z)
