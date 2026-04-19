from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device).float() / half
        )
        args = t * freqs.view(1, -1) * 2 * math.pi
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding


class VelocityMLP(nn.Module):
    def __init__(self, hidden: int = 256, t_dim: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(t_dim)
        in_dim = 2 + t_dim
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embedding(t)], dim=1))


class FlowMapMLP(nn.Module):
    def __init__(self, hidden: int = 256, d_dim: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.delta_embedding = SinusoidalTimeEmbedding(d_dim)
        in_dim = 2 + d_dim
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.delta_embedding(delta)], dim=1))


class NoisingMLP(nn.Module):
    def __init__(self, hidden: int = 256, r_dim: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(r_dim)
        in_dim = 2 + r_dim
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embedding(r)], dim=1))

