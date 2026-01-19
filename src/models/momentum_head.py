import torch
import torch.nn as nn


class MomentumHead(nn.Module):
    def __init__(
        self,
        env_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        """输出初始动量向量 (batch_size, out_dim)。"""
        if env.dim() == 1:
            env = env.unsqueeze(0)
        return self.net(env)
