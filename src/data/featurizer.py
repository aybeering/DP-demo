from typing import Dict, List

import torch
from rdkit import Chem
from torch_geometric.data import Data


def atom_features(atom: Chem.Atom) -> torch.Tensor:
    """单原子特征向量。"""
    features: List[float] = [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
    ]
    return torch.tensor(features, dtype=torch.float)


def bond_features(bond: Chem.Bond) -> torch.Tensor:
    """单键特征向量。"""
    bond_type = bond.GetBondType()
    features = [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    return torch.tensor(features, dtype=torch.float)


def mol_to_graph_data(
    mol: Chem.Mol,
    conf_id: int,
) -> Data:
    """RDKit 分子 + 构象 -> PyG Data。"""
    atoms = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atoms, dim=0)

    edge_index_list: List[List[int]] = []
    edge_attr_list: List[torch.Tensor] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = bond_features(bond)
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(bond_feat)
        edge_attr_list.append(bond_feat)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_list, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)

    conformer = mol.GetConformer(conf_id)
    positions = [
        [conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ]
    pos = torch.tensor(positions, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def build_feature_stats(dataset: list) -> Dict[str, float]:
    """统计特征维度、节点/边分布，用于日志。"""
    if not dataset:
        return {"num_samples": 0}
    num_nodes = [data.x.size(0) for data in dataset]
    num_edges = [data.edge_index.size(1) for data in dataset]
    return {
        "num_samples": float(len(dataset)),
        "avg_nodes": float(sum(num_nodes) / len(num_nodes)),
        "avg_edges": float(sum(num_edges) / len(num_edges)),
        "node_feature_dim": float(dataset[0].x.size(1)),
        "edge_feature_dim": float(dataset[0].edge_attr.size(1)) if dataset[0].edge_attr.numel() else 0.0,
    }
