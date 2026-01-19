import torch
import torch.nn as nn


class StepPredictor(nn.Module):
    def __init__(
        self,
        z_dim: int,
        env_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """输出 delta_z (batch_size, z_dim)。"""
        if env.dim() == 1:
            env = env.unsqueeze(0).expand(z.size(0), -1)
        inputs = torch.cat([z, env], dim=-1)
        return self.net(inputs)
