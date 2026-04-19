from __future__ import annotations

import math

import torch

from .config import ToyConfig


def ring_centers(config: ToyConfig, device: torch.device) -> torch.Tensor:
    angles = torch.linspace(0, 2 * math.pi, config.k + 1, device=device)[:-1]
    return torch.stack(
        [config.ring_radius * torch.cos(angles), config.ring_radius * torch.sin(angles)],
        dim=1,
    )


def sample_true(config: ToyConfig, centers: torch.Tensor, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    modes = torch.randint(0, config.k, (n,), device=device)
    x = centers[modes] + config.data_sigma * torch.randn(n, 2, device=device)
    return x, modes


def sample_mismatched(
    config: ToyConfig,
    centers: torch.Tensor,
    n: int,
    strength: float,
    device: torch.device,
) -> torch.Tensor:
    strength = float(strength)
    drop = int(round(2 * strength))
    keep_k = max(2, config.k - drop)
    keep_centers = centers[:keep_k]

    modes = torch.randint(0, keep_k, (n,), device=device)
    x = keep_centers[modes] + config.data_sigma * torch.randn(n, 2, device=device)

    theta = (math.pi / 3) * strength
    rotation = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        device=device,
        dtype=x.dtype,
    )
    x = x @ rotation.T
    x = x + (0.25 * strength) * torch.randn_like(x)
    return x


def log_prob_true(config: ToyConfig, centers: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(1)
    centers = centers.unsqueeze(0)
    diff2 = ((x - centers) ** 2).sum(dim=-1)
    log_component = -0.5 * diff2 / (config.data_sigma**2) - math.log(2 * math.pi * (config.data_sigma**2))
    return torch.logsumexp(log_component - math.log(config.k), dim=1)

