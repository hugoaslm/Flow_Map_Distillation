from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def save_scatter(path: str | Path, samples: torch.Tensor, centers: torch.Tensor, title: str) -> None:
    arr = _to_numpy(samples)
    ctr = _to_numpy(centers)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(arr[:, 0], arr[:, 1], s=2, alpha=0.4)
    ax.scatter(ctr[:, 0], ctr[:, 1], s=120, marker="x", color="orange")
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_metric_curves(
    path: str | Path,
    nfes: tuple[int, ...],
    teacher_rows: list[dict],
    baseline_rows: list[dict],
    freeflow_row: dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric in zip(axes, ("nll", "swd"), strict=True):
        axis.plot(nfes, [row[metric] for row in teacher_rows], marker="o", label="teacher")
        axis.plot(nfes, [row[metric] for row in baseline_rows], marker="o", label="vel-clone")
        axis.scatter([1], [freeflow_row[metric]], marker="x", s=120, label="freeflow")
        axis.set_xlabel("NFE")
        axis.set_ylabel(metric)
        axis.set_title(f"{metric} vs NFE")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

