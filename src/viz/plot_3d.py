from typing import List

import matplotlib.pyplot as plt
import torch


def project_to_3d(z_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """PCA 降维到 3D（如需要）。"""
    if not z_list:
        return []
    z = torch.stack(z_list, dim=0)
    if z.size(1) <= 3:
        return [item for item in z]

    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = z_centered.t().mm(z_centered) / (z_centered.size(0) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    topk = eigvecs[:, -3:]
    projected = z_centered.mm(topk)
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
