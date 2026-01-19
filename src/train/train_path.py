import os

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.data.conformer_gen import build_conformer_dataset
from src.data.dataset import PathDataset, build_path_samples
from src.models.energy_head import EnergyHead
from src.models.gnn_encoder import GNNEncoder
from src.models.step_predictor import StepPredictor
from src.train.losses import total_loss
from src.utils.config import TrainConfig
from src.utils.logging import setup_logger


def train_one_epoch(
    encoder,
    stepper,
    energy_head,
    loader: DataLoader,
    optimizer,
    device: torch.device,
) -> float:
    """单个 epoch 训练，返回平均 loss。"""
    encoder.train()
    stepper.train()
    energy_head.train()

    total = 0.0
    for batch in loader:
        data_t = batch["data_t"].to(device)
        data_next = batch["data_next"].to(device)
        env = batch["env"].to(device)
        energy_true = batch["energy_next"].to(device)

        z_t = encoder(data_t)
        z_next_true = encoder(data_next).detach()
        delta_z = stepper(z_t, env)
        z_next_pred = z_t + delta_z
        energy_pred = energy_head(z_next_pred)

        loss = total_loss(z_next_pred, z_next_true, energy_pred, energy_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / max(len(loader), 1)


def train(
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> None:
    """训练入口。"""
    logger = setup_logger("train")
    config = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr)

    os.makedirs("outputs", exist_ok=True)

    conformer_data = build_conformer_dataset(config.smiles, num_confs=config.num_confs)
    samples = build_path_samples(conformer_data, env={"temperature": 300.0})
    dataset = PathDataset(samples)

    loader = GeoDataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = samples[0]["data_t"].x.size(1)
    env_dim = samples[0]["env"].numel()

    encoder = GNNEncoder(in_dim=in_dim, out_dim=config.emb_dim).to(device)
    stepper = StepPredictor(z_dim=config.emb_dim, env_dim=env_dim).to(device)
    energy_head = EnergyHead(in_dim=config.emb_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(stepper.parameters()) + list(energy_head.parameters()),
        lr=config.lr,
    )

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(encoder, stepper, energy_head, loader, optimizer, device)
        logger.info("Epoch %s | loss=%.4f", epoch + 1, avg_loss)

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "stepper": stepper.state_dict(),
            "energy_head": energy_head.state_dict(),
        },
        "outputs/model.pt",
    )
