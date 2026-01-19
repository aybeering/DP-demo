from typing import List, Tuple

import torch


def simulate_path(
    z0: torch.Tensor,
    env: torch.Tensor,
    stepper,
    energy_head,
    num_steps: int = 20,
    step_size: float = 1.0,
) -> Tuple[List[torch.Tensor], List[float]]:
    """返回 z 轨迹与能量序列。"""
    z_list = [z0]
    energies = [float(energy_head(z0).item())]

    z = z0
    for _ in range(num_steps):
        delta_z = stepper(z, env)
        z = z + step_size * delta_z
        z_list.append(z)
        energies.append(float(energy_head(z).item()))

    return z_list, energies
