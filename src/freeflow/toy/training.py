from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .config import ToyConfig


def sample_rf_batch(
    config: ToyConfig,
    x0_sampler: Callable[[int], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = x0_sampler(config.batch_size).to(device)
    z = torch.randn(config.batch_size, 2, device=device)
    t = torch.rand(config.batch_size, 1, device=device) * (1 - 2 * config.t_min) + config.t_min
    x_t = (1 - t) * x0 + t * z
    v = x0 - z
    return x_t, t, v


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    n: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device).view(-1, 1)
    x = torch.randn(n, 2, device=device)
    for i in range(steps):
        t_curr = ts[i].expand(n, 1)
        t_next = ts[i + 1].expand(n, 1)
        x = x - (t_next - t_curr) * model(x, t_curr)
    return x


def train_teacher(
    model: nn.Module,
    config: ToyConfig,
    device: torch.device,
    x0_sampler: Callable[[int], torch.Tensor],
) -> nn.Module:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_teacher, weight_decay=1e-4)
    for step in tqdm(range(config.teacher_steps), desc="Teacher train"):
        x_t, t, target = sample_rf_batch(config, x0_sampler, device)
        prediction = model(x_t, t)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    model.eval()
    return model


def simulate_student_trajectory(
    student: nn.Module,
    z: torch.Tensor,
    time_horizon: torch.Tensor,
    steps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = z
    for step in range(steps):
        t_curr = 1.0 - (step / steps) * time_horizon
        t_next = 1.0 - ((step + 1) / steps) * time_horizon
        x = x - (t_next - t_curr) * student(x, t_curr)
    return x, 1.0 - time_horizon


def train_student_vel_clone(
    student: nn.Module,
    teacher: nn.Module,
    config: ToyConfig,
    device: torch.device,
    sim_steps: int = 8,
    anchor_weight: float = 0.1,
) -> nn.Module:
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr_student, weight_decay=1e-4)
    ones_cache: torch.Tensor | None = None
    for _ in tqdm(range(config.student_steps), desc="Velocity-clone baseline"):
        n = config.batch_size
        z = torch.randn(n, 2, device=device)
        horizon = torch.rand(n, 1, device=device) * (1 - 2 * config.t_min) + config.t_min
        x, t_end = simulate_student_trajectory(student, z, horizon, steps=sim_steps)
        with torch.no_grad():
            teacher_velocity = teacher(x, t_end)
        student_velocity = student(x, t_end)
        loss = F.mse_loss(student_velocity, teacher_velocity)

        if anchor_weight > 0:
            if ones_cache is None or ones_cache.shape[0] != n:
                ones_cache = torch.ones(n, 1, device=device)
            with torch.no_grad():
                teacher_anchor = teacher(z, ones_cache)
            loss = loss + anchor_weight * F.mse_loss(student(z, ones_cache), teacher_anchor)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
        optimizer.step()
    student.eval()
    return student


def dF_d_delta_jvp(student_f: nn.Module, z: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.detach()
    delta = delta.requires_grad_(True)

    def f_of_delta(value: torch.Tensor) -> torch.Tensor:
        return student_f(z, value)

    tangent = torch.ones_like(delta)
    return torch.autograd.functional.jvp(f_of_delta, (delta,), (tangent,), create_graph=True)


def generating_velocity(student_f: nn.Module, z: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f_avg, d_f = dF_d_delta_jvp(student_f, z, delta)
    flow = z + delta * f_avg
    return flow, 1.0 - delta, f_avg, f_avg + delta * d_f


@torch.no_grad()
def sample_one_step_freeflow(student_f: nn.Module, n: int, device: torch.device) -> torch.Tensor:
    z = torch.randn(n, 2, device=device)
    ones = torch.ones(n, 1, device=device)
    return z + student_f(z, ones)


@torch.no_grad()
def warmup_renoise(x_t: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, tc: torch.Tensor) -> torch.Tensor:
    mask = tc > t
    a = (1.0 - tc) / (1.0 - t + 1e-12)
    rad = tc**2 - (a**2) * (t**2)
    b = torch.sqrt(torch.clamp(rad, min=0.0))
    transformed = a * x_t + b * noise
    return torch.where(mask, transformed, x_t)


def train_student_freeflow(
    student_f: nn.Module,
    g_noising: nn.Module,
    teacher: nn.Module,
    config: ToyConfig,
    device: torch.device,
    alpha: float = 0.1,
    warmup_steps: int | None = None,
) -> tuple[nn.Module, nn.Module]:
    if warmup_steps is None:
        warmup_steps = min(1000, config.student_steps // 2)

    student_f.train()
    g_noising.train()
    opt_f = torch.optim.AdamW(student_f.parameters(), lr=config.lr_student, weight_decay=1e-4)
    opt_g = torch.optim.AdamW(g_noising.parameters(), lr=config.lr_student * 2.0, weight_decay=1e-4)
    ones_cache: torch.Tensor | None = None

    for step in tqdm(range(config.student_steps), desc="FreeFlow train"):
        n = config.batch_size
        z = torch.randn(n, 2, device=device)
        delta = torch.rand(n, 1, device=device) * (1 - 2 * config.t_min) + config.t_min
        flow, t, f_avg, v_g = generating_velocity(student_f, z, delta)

        with torch.no_grad():
            if step < warmup_steps:
                frac = step / max(1, warmup_steps - 1)
                tc = torch.full_like(t, 1.0 - frac)
                flow_for_teacher = warmup_renoise(flow, torch.randn_like(flow), t, tc)
                u = teacher(flow_for_teacher, tc)
            else:
                u = teacher(flow, t)

        loss_pred = (f_avg * (v_g - u).detach()).sum(dim=1).mean()

        if ones_cache is None or ones_cache.shape[0] != n:
            ones_cache = torch.ones(n, 1, device=device)
        x0 = z + student_f(z, ones_cache)

        noise = torch.randn(n, 2, device=device)
        r = torch.rand(n, 1, device=device) * (1 - 2 * config.t_min) + config.t_min
        xr = (1 - r) * x0.detach() + r * noise
        d_ir = noise - x0.detach()

        loss_g = F.mse_loss(g_noising(xr, r), -d_ir)
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        nn.utils.clip_grad_norm_(g_noising.parameters(), config.grad_clip)
        opt_g.step()

        with torch.no_grad():
            u_r = teacher(xr, r)
        correction = (g_noising(xr, r) - u_r).detach()
        loss_corr = (student_f(z, ones_cache) * correction).sum(dim=1).mean()
        loss = loss_pred + alpha * loss_corr

        opt_f.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student_f.parameters(), config.grad_clip)
        opt_f.step()

    student_f.eval()
    g_noising.eval()
    return student_f, g_noising


def train_student_data_based(
    student: nn.Module,
    teacher: nn.Module,
    config: ToyConfig,
    device: torch.device,
    x0_sampler: Callable[[int], torch.Tensor],
) -> nn.Module:
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr_student, weight_decay=1e-4)
    for _ in tqdm(range(config.student_steps), desc="Data-based student"):
        x_t, t, _ = sample_rf_batch(config, x0_sampler, device)
        with torch.no_grad():
            teacher_velocity = teacher(x_t, t)
        loss = F.mse_loss(student(x_t, t), teacher_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
        optimizer.step()
    student.eval()
    return student

