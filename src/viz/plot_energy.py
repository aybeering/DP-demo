from typing import List

import matplotlib.pyplot as plt


def plot_energy_curve(energies: List[float], out_path: str) -> None:
    """绘制能量曲线并保存。"""
    plt.figure(figsize=(6, 4))
    plt.plot(energies, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
