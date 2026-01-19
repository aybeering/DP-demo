import argparse
import os
import sys

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.conformer_gen import build_conformer_dataset
from src.data.dataset import PathDataset, build_path_samples
from src.models.energy_head import EnergyHead
from src.models.gnn_encoder import GNNEncoder
from src.models.momentum_head import MomentumHead
from src.models.step_predictor import StepPredictor
from src.sim.simulate import simulate_path
from src.train.train_path import train_one_epoch
from src.utils.logging import setup_logger
from src.viz.plot_3d import plot_3d_path, plot_energy_terrain
from src.viz.plot_energy import plot_energy_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DP-demo MVP runner")
    parser.add_argument("--smiles", type=str, default="CCCC")
    parser.add_argument("--num-confs", type=int, default=20)
    parser.add_argument("--energy-method", type=str, default="MMFF", choices=["MMFF", "UFF"])
    parser.add_argument("--temperature", type=float, default=300.0)

    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--sim-steps", type=int, default=20)
    parser.add_argument("--sim-step-size", type=float, default=1.0)
    parser.add_argument("--field-scale", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("mvp")

    os.makedirs(args.outdir, exist_ok=True)

    conformer_data = build_conformer_dataset(
        args.smiles,
        num_confs=args.num_confs,
        method=args.energy_method,
    )
    samples = build_path_samples(conformer_data, env={"temperature": args.temperature})
    if not samples:
        raise RuntimeError("No training samples generated (need >= 2 conformers).")

    dataset = PathDataset(samples)
    loader = GeoDataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = samples[0]["data_t"].x.size(1)
    env_dim = samples[0]["env"].numel()

    encoder = GNNEncoder(in_dim=in_dim, out_dim=args.emb_dim).to(device)
    stepper = StepPredictor(z_dim=args.emb_dim, env_dim=env_dim).to(device)
    momentum_head = MomentumHead(env_dim=env_dim, out_dim=args.emb_dim).to(device)
    energy_head = EnergyHead(in_dim=args.emb_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(stepper.parameters())
        + list(momentum_head.parameters())
        + list(energy_head.parameters()),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            encoder,
            stepper,
            momentum_head,
            energy_head,
            loader,
            optimizer,
            device,
            field_scale=args.field_scale,
        )
        logger.info("Epoch %s | loss=%.4f", epoch + 1, avg_loss)

    model_path = os.path.join(args.outdir, "model.pt")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "stepper": stepper.state_dict(),
            "momentum_head": momentum_head.state_dict(),
            "energy_head": energy_head.state_dict(),
        },
        model_path,
    )

    encoder.eval()
    stepper.eval()
    energy_head.eval()
    with torch.no_grad():
        data0 = samples[0]["data_t"].to(device)
        env0 = samples[0]["env"].to(device)
        z0 = encoder(data0)
        z_list, energies = simulate_path(
            z0,
            env0,
            stepper,
            energy_head,
            momentum_head=momentum_head,
            num_steps=args.sim_steps,
            step_size=args.sim_step_size,
            field_scale=args.field_scale,
        )

    z_list = [z.squeeze(0).cpu() for z in z_list]
    energy_plot_path = os.path.join(args.outdir, "energy_curve.png")
    path_plot_path = os.path.join(args.outdir, "path_3d.png")
    terrain_plot_path = os.path.join(args.outdir, "energy_terrain.png")
    plot_energy_curve(energies, energy_plot_path)
    plot_3d_path(z_list, path_plot_path)
    plot_energy_terrain(z_list, energy_head, terrain_plot_path)

    print("Wrote:")
    print(f"- {model_path}")
    print(f"- {energy_plot_path}")
    print(f"- {path_plot_path}")
    print(f"- {terrain_plot_path}")


if __name__ == "__main__":
    main()
