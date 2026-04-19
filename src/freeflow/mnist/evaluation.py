from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import MnistConfig
from .data import MnistDataModule
from .models import TinyEncoder
from .training import euler_sample_images, sample_one_step_freeflow


@torch.no_grad()
def sliced_wasserstein_images(x: torch.Tensor, y: torch.Tensor, num_proj: int = 128) -> float:
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    proj = torch.randn(num_proj, x.shape[1], device=x.device)
    proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-12)
    x_sorted, _ = torch.sort(x @ proj.T, dim=0)
    y_sorted, _ = torch.sort(y @ proj.T, dim=0)
    return torch.mean(torch.abs(x_sorted - y_sorted)).item()


@torch.no_grad()
def sliced_wasserstein_features(
    encoder: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_proj: int = 128,
) -> float:
    fx = encoder(x)
    fy = encoder(y)
    proj = torch.randn(num_proj, fx.shape[1], device=fx.device)
    proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-12)
    x_sorted, _ = torch.sort(fx @ proj.T, dim=0)
    y_sorted, _ = torch.sort(fy @ proj.T, dim=0)
    return torch.mean(torch.abs(x_sorted - y_sorted)).item()


@torch.no_grad()
def evaluate_models_vs_nfe(
    teacher: nn.Module,
    freeflow_f: nn.Module,
    config: MnistConfig,
    data: MnistDataModule,
    device: torch.device,
    nfes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    use_feature_swd: bool = True,
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor], torch.Tensor]:
    real = data.sample_real_images(config.n_eval).clamp(-1, 1)
    encoder = TinyEncoder(in_ch=config.channels, feat_dim=128).to(device).eval() if use_feature_swd else None
    rows: list[dict[str, Any]] = []
    teacher_samples: dict[int, torch.Tensor] = {}

    for nfe in nfes:
        xt = euler_sample_images(teacher, config, config.n_eval, steps=nfe, device=device).clamp(-1, 1)
        metric = (
            sliced_wasserstein_features(encoder, xt, real)
            if encoder is not None
            else sliced_wasserstein_images(xt, real)
        )
        rows.append({"model": "teacher", "nfe": nfe, "swd": metric})
        teacher_samples[nfe] = xt

    xf = sample_one_step_freeflow(freeflow_f, config, config.n_eval, device=device).clamp(-1, 1)
    free_metric = (
        sliced_wasserstein_features(encoder, xf, real)
        if encoder is not None
        else sliced_wasserstein_images(xf, real)
    )
    rows.append({"model": "freeflow", "nfe": 1, "swd": free_metric})
    return rows, teacher_samples, xf

