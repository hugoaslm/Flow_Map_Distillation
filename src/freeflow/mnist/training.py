from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .config import MnistConfig
from .data import MnistDataModule


def sample_rf_batch_images(
    config: MnistConfig,
    data: MnistDataModule,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = data.sample_real_images(config.batch_size)
    z = torch.randn_like(x0)
    t = torch.rand(config.batch_size, 1, device=x0.device) * (1 - 2 * config.t_min) + config.t_min
    x_t = (1 - t)[:, :, None, None] * x0 + t[:, :, None, None] * z
    return x_t, t, x0 - z


@torch.no_grad()
def euler_sample_images(
    model: nn.Module,
    config: MnistConfig,
    n: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device).view(-1, 1)
    x = torch.randn(n, config.channels, config.img_size, config.img_size, device=device)
    for i in range(steps):
        t_curr = ts[i].expand(n, 1)
        t_next = ts[i + 1].expand(n, 1)
        x = x - (t_next - t_curr)[:, :, None, None] * model(x, t_curr)
    return x


def train_teacher(
    model: nn.Module,
    config: MnistConfig,
    data: MnistDataModule,
) -> nn.Module:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_teacher, weight_decay=1e-4)
    for _ in tqdm(range(config.teacher_steps), desc="Teacher train (MNIST)"):
        x_t, t, target = sample_rf_batch_images(config, data)
        loss = F.mse_loss(model(x_t, t), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    model.eval()
    return model


def dF_d_delta_jvp(student_f: nn.Module, z: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.detach()
    delta = delta.requires_grad_(True)

    def f_of_delta(value: torch.Tensor) -> torch.Tensor:
        return student_f(z, 1.0 - value)

    tangent = torch.ones_like(delta)
    return torch.autograd.functional.jvp(f_of_delta, (delta,), (tangent,), create_graph=True)


def generating_velocity(student_f: nn.Module, z: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f_avg, d_f = dF_d_delta_jvp(student_f, z, delta)
    flow = z + delta[:, :, None, None] * f_avg
    return flow, 1.0 - delta, f_avg, f_avg + delta[:, :, None, None] * d_f


@torch.no_grad()
def sample_one_step_freeflow(
    student_f: nn.Module,
    config: MnistConfig,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    z = torch.randn(n, config.channels, config.img_size, config.img_size, device=device)
    zeros = torch.zeros(n, 1, device=device)
    return z + student_f(z, zeros)


@torch.no_grad()
def warmup_renoise(x_t: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, tc: torch.Tensor) -> torch.Tensor:
    mask = tc > t
    a = (1.0 - tc) / (1.0 - t + 1e-12)
    rad = tc**2 - (a**2) * (t**2)
    b = torch.sqrt(torch.clamp(rad, min=0.0))
    transformed = a[:, :, None, None] * x_t + b[:, :, None, None] * noise
    return torch.where(mask[:, :, None, None], transformed, x_t)


def train_student_freeflow(
    student_f: nn.Module,
    g_noising: nn.Module,
    teacher: nn.Module,
    config: MnistConfig,
    device: torch.device,
    alpha: float = 0.1,
    warmup_steps: int | None = None,
) -> tuple[nn.Module, nn.Module]:
    if warmup_steps is None:
        warmup_steps = min(2000, config.student_steps // 2)

    student_f.train()
    g_noising.train()
    opt_f = torch.optim.AdamW(student_f.parameters(), lr=config.lr_student, weight_decay=1e-4)
    opt_g = torch.optim.AdamW(g_noising.parameters(), lr=config.lr_student * 2.0, weight_decay=1e-4)
    ones_cache: torch.Tensor | None = None

    for step in tqdm(range(config.student_steps), desc="FreeFlow train (MNIST)"):
        n = config.batch_size
        z = torch.randn(n, config.channels, config.img_size, config.img_size, device=device)
        delta = torch.rand(n, 1, device=device) * (1 - 2 * config.t_min) + config.t_min
        flow, t, f_avg, v_g = generating_velocity(student_f, z, delta)

        with torch.no_grad():
            if step < warmup_steps:
                frac = step / max(1, warmup_steps - 1)
                tc = torch.full_like(t, 1.0 - frac)
                teacher_input = warmup_renoise(flow, torch.randn_like(flow), t, tc)
                u = teacher(teacher_input, tc)
            else:
                u = teacher(flow, t)

        loss_pred = (f_avg * (v_g - u).detach()).mean()

        if ones_cache is None or ones_cache.shape[0] != n:
            ones_cache = torch.ones(n, 1, device=device)
        f1 = student_f(z, ones_cache)
        x0 = z + ones_cache[:, :, None, None] * f1

        noise = torch.randn_like(z)
        r = torch.rand(n, 1, device=device) * (1 - 2 * config.t_min) + config.t_min
        xr = (1 - r)[:, :, None, None] * x0.detach() + r[:, :, None, None] * noise
        d_ir = noise - x0.detach()

        loss_g = F.mse_loss(g_noising(xr, r), -d_ir)
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        nn.utils.clip_grad_norm_(g_noising.parameters(), config.grad_clip)
        opt_g.step()

        with torch.no_grad():
            u_r = teacher(xr, r)
        loss_corr = (f1 * (g_noising(xr, r) - u_r).detach()).mean()
        loss = loss_pred + alpha * loss_corr

        opt_f.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student_f.parameters(), config.grad_clip)
        opt_f.step()

    student_f.eval()
    g_noising.eval()
    return student_f, g_noising
