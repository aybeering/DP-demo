from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.featurizer import mol_to_graph_data


class PathDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
    ) -> None:
        """samples: [{"data_t": Data, "data_next": Data, "env": tensor, ...}, ...]"""
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def build_path_samples(
    conformer_data: List[Tuple["Chem.Mol", int, float]],
    env: Dict[str, float],
) -> List[Dict[str, Any]]:
    """将构象序列组织成训练样本。"""
    samples: List[Dict[str, Any]] = []
    if len(conformer_data) < 2:
        return samples

    env_tensor = torch.tensor(list(env.values()), dtype=torch.float)
    for (mol_t, conf_t, energy_t), (mol_next, conf_next, energy_next) in zip(
        conformer_data[:-1], conformer_data[1:]
    ):
        data_t = mol_to_graph_data(mol_t, conf_t)
        data_next = mol_to_graph_data(mol_next, conf_next)
        samples.append(
            {
                "data_t": data_t,
                "data_next": data_next,
                "env": env_tensor,
                "energy_t": torch.tensor([energy_t], dtype=torch.float),
                "energy_next": torch.tensor([energy_next], dtype=torch.float),
            }
        )
    return samples
