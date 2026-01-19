import torch


def path_loss(z_pred_next: torch.Tensor, z_true_next: torch.Tensor) -> torch.Tensor:
    """路径回归损失。"""
    return torch.mean((z_pred_next - z_true_next) ** 2)


def energy_loss(energy_pred: torch.Tensor, energy_true: torch.Tensor) -> torch.Tensor:
    """能量回归损失。"""
    return torch.mean((energy_pred - energy_true) ** 2)


def total_loss(
    z_pred_next: torch.Tensor,
    z_true_next: torch.Tensor,
    energy_pred: torch.Tensor,
    energy_true: torch.Tensor,
    lambda_energy: float = 0.1,
) -> torch.Tensor:
    """组合损失。"""
    return path_loss(z_pred_next, z_true_next) + lambda_energy * energy_loss(energy_pred, energy_true)
