from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


def build_molecule(smiles: str) -> Chem.Mol:
    """从 SMILES 构建 RDKit 分子对象（含氢）。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.AddHs(mol)


def generate_conformers(
    mol: Chem.Mol,
    num_confs: int = 20,
    seed: int = 42,
) -> List[int]:
    """生成构象并返回 conformer ids。"""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    return conf_ids


def compute_conformer_energy(
    mol: Chem.Mol,
    conf_id: int,
    method: str = "MMFF",
) -> float:
    """计算指定构象的力场能量。"""
    method = method.upper()
    if method == "MMFF":
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is None:
            raise ValueError("MMFF properties unavailable for molecule")
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    elif method == "UFF":
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    else:
        raise ValueError(f"Unsupported method: {method}")
    if ff is None:
        raise ValueError("Failed to build force field")
    return float(ff.CalcEnergy())


def build_conformer_dataset(
    smiles: str,
    num_confs: int = 20,
    method: str = "MMFF",
) -> List[Tuple[Chem.Mol, int, float]]:
    """返回 [(mol, conf_id, energy), ...]。"""
    mol = build_molecule(smiles)
    conf_ids = generate_conformers(mol, num_confs=num_confs)
    dataset = []
    for conf_id in conf_ids:
        energy = compute_conformer_energy(mol, conf_id, method=method)
        dataset.append((mol, conf_id, energy))
    return dataset
