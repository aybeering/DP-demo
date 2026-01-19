import torch
import torch.nn as nn


class MomentumHead(nn.Module):
    def __init__(self, env_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, out_dim),
        )

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        """输出初始动量向量 (batch_size, out_dim)。"""
        return self.net(env)
