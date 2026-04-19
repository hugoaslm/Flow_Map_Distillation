from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import ToyConfig
from .data import log_prob_true, sample_true
from .training import euler_sample, sample_one_step_freeflow


@torch.no_grad()
def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, num_proj: int = 128) -> float:
    x = x[:4000]
    y = y[:4000]
    proj = torch.randn(num_proj, x.shape[1], device=x.device)
    proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
    x_proj, _ = torch.sort(x @ proj.T, dim=0)
    y_proj, _ = torch.sort(y @ proj.T, dim=0)
    length = min(x_proj.shape[0], y_proj.shape[0])
    return torch.mean(torch.abs(x_proj[:length] - y_proj[:length])).item()


@torch.no_grad()
def evaluate_velocity_model(
    model: nn.Module,
    config: ToyConfig,
    device: torch.device,
    centers: torch.Tensor,
    steps: int,
    n_gen: int,
) -> dict[str, Any]:
    generated = euler_sample(model, n_gen, steps=steps, device=device)
    real, _ = sample_true(config, centers, n_gen, device)
    return {
        "nfe": steps,
        "swd": sliced_wasserstein(generated, real),
        "nll": (-log_prob_true(config, centers, generated)).mean().item(),
        "gen": generated,
    }


@torch.no_grad()
def evaluate_flowmap_model(
    model: nn.Module,
    config: ToyConfig,
    device: torch.device,
    centers: torch.Tensor,
    n_gen: int,
) -> dict[str, Any]:
    generated = sample_one_step_freeflow(model, n_gen, device=device)
    real, _ = sample_true(config, centers, n_gen, device)
    return {
        "nfe": 1,
        "swd": sliced_wasserstein(generated, real),
        "nll": (-log_prob_true(config, centers, generated)).mean().item(),
        "gen": generated,
    }

