from typing import List

import matplotlib.pyplot as plt
import torch.nn as nn
import torch


def _infer_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _pca_project(
    z_list: List[torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not z_list:
        raise ValueError("z_list is empty")

    z = torch.stack(z_list, dim=0)
    if z.dim() != 2:
        raise ValueError(f"Expected 2D stacked z, got shape={tuple(z.shape)}")

    z_dim = z.size(1)
    k = min(k, z_dim)
    mean = z.mean(dim=0, keepdim=True)
    z_centered = z - mean

    if k == z_dim:
        basis = torch.eye(z_dim, device=z.device, dtype=z.dtype)
        projected = z_centered
        return projected, mean.squeeze(0), basis

    denom = max(z_centered.size(0) - 1, 1)
    cov = z_centered.t().mm(z_centered) / denom
    _, eigvecs = torch.linalg.eigh(cov)
    basis = eigvecs[:, -k:]
    projected = z_centered.mm(basis)
    return projected, mean.squeeze(0), basis


def project_to_3d(z_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """PCA 降维到 3D（如需要）。"""
    if not z_list:
        return []
    z = torch.stack(z_list, dim=0)
    if z.size(1) <= 3:
        return [item for item in z]

    projected, _, _ = _pca_project(z_list, k=3)
    return [item for item in projected]


def plot_3d_path(z_list: List[torch.Tensor], out_path: str) -> None:
    """绘制 3D 轨迹图。"""
    projected = project_to_3d(z_list)
    if not projected:
        return
    coords = torch.stack(projected, dim=0).cpu().numpy()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_energy_terrain(
    z_list: List[torch.Tensor],
    energy_head: nn.Module,
    out_path: str,
    grid_size: int = 80,
    padding: float = 0.2,
) -> None:
    """绘制类似“热力地形图”的能量地貌，并叠加轨迹。

    - x/y: 轨迹在 PCA(2D) 平面上的坐标
    - z(height) / colormap: energy_head(z) 预测能量
    """
    if not z_list:
        return
    if grid_size < 10:
        raise ValueError("grid_size must be >= 10")

    device = _infer_device(energy_head)
    z_list_dev = [z.detach().to(device) for z in z_list]

    z_dim = z_list_dev[0].numel()
    k = 2 if z_dim >= 2 else 1
    projected, mean, basis = _pca_project(z_list_dev, k=k)
    x_path = projected[:, 0].detach().cpu()
    if k == 2:
        y_path = projected[:, 1].detach().cpu()
    else:
        y_path = torch.zeros_like(x_path)

    x_min, x_max = float(x_path.min().item()), float(x_path.max().item())
    y_min, y_max = float(y_path.min().item()), float(y_path.max().item())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_min -= padding * x_span
    x_max += padding * x_span
    y_min -= padding * y_span
    y_max += padding * y_span

    xs = torch.linspace(x_min, x_max, grid_size, device=device)
    ys = torch.linspace(y_min, y_max, grid_size, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    if k == 2:
        grid_xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # (N, 2)
        z_hat = mean.unsqueeze(0) + grid_xy.mm(basis.t())
    else:
        grid_x = X.reshape(-1).unsqueeze(1)  # (N, 1)
        z_hat = mean.unsqueeze(0) + grid_x.mm(basis.t())
    with torch.no_grad():
        energy_hat = energy_head(z_hat).reshape(grid_size, grid_size)
        energy_path = energy_head(torch.stack(z_list_dev, dim=0)).squeeze(-1)

    X_cpu = X.detach().cpu().numpy()
    Y_cpu = Y.detach().cpu().numpy()
    E_cpu = energy_hat.detach().cpu().numpy()
    E_path_cpu = energy_path.detach().cpu().numpy()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X_cpu, Y_cpu, E_cpu, cmap="terrain", linewidth=0, antialiased=True, alpha=0.95)

    z_offset = float(E_cpu.min())
    ax.contourf(X_cpu, Y_cpu, E_cpu, zdir="z", offset=z_offset, cmap="terrain", alpha=0.6)

    ax.plot(
        x_path.numpy(),
        y_path.numpy(),
        E_path_cpu,
        color="black",
        linewidth=2.0,
        marker="o",
        markersize=3,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Energy")
    ax.view_init(elev=45, azim=-60)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
