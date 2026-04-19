from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def save_sample_grid(path: str | Path, x: torch.Tensor, title: str, nrow: int = 8) -> None:
    x = x.clamp(-1, 1)
    grid = make_grid((x + 1) / 2, nrow=nrow)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_swd_curve(path: str | Path, rows: list[dict], nfes: tuple[int, ...]) -> None:
    teacher = [row["swd"] for row in rows if row["model"] == "teacher"]
    freeflow = next(row["swd"] for row in rows if row["model"] == "freeflow")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(nfes), teacher, marker="o", label="teacher")
    ax.scatter([1], [freeflow], marker="x", s=120, label="freeflow")
    ax.set_xlabel("NFE")
    ax.set_ylabel("SWD")
    ax.set_title("Quality vs NFE (MNIST)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

