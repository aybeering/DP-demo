import unittest

import torch
import torch.nn as nn

from src.sim.simulate import simulate_path


class ZeroStepper(nn.Module):
    def forward(self, z: torch.Tensor, env: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return torch.zeros_like(z)


class ConstantField(nn.Module):
    def __init__(self, delta: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("delta", delta)

    def forward(self, env: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        if env.dim() == 1:
            env = env.unsqueeze(0)
        return self.delta.unsqueeze(0).expand(env.size(0), -1)


class SumEnergy(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return z.sum(dim=-1, keepdim=True)


class TestExternalFieldDriving(unittest.TestCase):
    def test_simulate_path_applies_field_each_step(self) -> None:
        z0 = torch.zeros(1, 3)
        env = torch.tensor([300.0])
        stepper = ZeroStepper()
        energy_head = SumEnergy()
        momentum_head = ConstantField(delta=torch.tensor([1.0, 2.0, 3.0]))

        z_list, _energies = simulate_path(
            z0,
            env,
            stepper=stepper,
            energy_head=energy_head,
            momentum_head=momentum_head,
            num_steps=2,
            step_size=1.0,
            field_scale=0.5,
        )

        self.assertEqual(len(z_list), 3)
        self.assertTrue(torch.allclose(z_list[0], z0))
        self.assertTrue(torch.allclose(z_list[1], torch.tensor([[0.5, 1.0, 1.5]])))
        self.assertTrue(torch.allclose(z_list[2], torch.tensor([[1.0, 2.0, 3.0]])))


if __name__ == "__main__":
    unittest.main()

