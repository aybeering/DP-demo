from dataclasses import dataclass


@dataclass
class TrainConfig:
    smiles: str = "CCCC"
    num_confs: int = 20
    emb_dim: int = 16
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-3
