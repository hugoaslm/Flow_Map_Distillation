from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128, max_period: float = 10000.0) -> None:
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


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetLite(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 64, temb_dim: int = 128) -> None:
        super().__init__()
        self.temb = SinusoidalTimeEmbedding(temb_dim)
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock(base, base, temb_dim)
        self.down1 = Down(base)
        self.rb2 = ResBlock(base, base, temb_dim)
        self.down2 = Down(base)
        self.mid = ResBlock(base, base, temb_dim)
        self.up2 = Up(base)
        self.rb3 = ResBlock(base + base, base, temb_dim)
        self.up1 = Up(base)
        self.rb4 = ResBlock(base + base, base, temb_dim)
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb(t)
        x0 = self.in_conv(x)
        h1 = self.rb1(x0, temb)
        h2 = self.rb2(self.down1(h1), temb)
        mid = self.mid(self.down2(h2), temb)
        u2 = self.rb3(torch.cat([self.up2(mid), h2], dim=1), temb)
        u1 = self.rb4(torch.cat([self.up1(u2), h1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(u1)))


class TinyEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, feat_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)

