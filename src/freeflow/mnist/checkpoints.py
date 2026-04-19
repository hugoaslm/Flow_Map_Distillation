from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    teacher: nn.Module,
    student_f: nn.Module,
    g_noising: nn.Module,
    config: Any | None = None,
) -> None:
    payload = {
        "teacher": teacher.state_dict(),
        "student_F": student_f.state_dict(),
        "g_noising": g_noising.state_dict(),
    }
    if config is not None:
        payload["config"] = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    teacher: nn.Module,
    student_f: nn.Module,
    g_noising: nn.Module,
    map_location=None,
    strict: bool = True,
) -> dict:
    payload = torch.load(path, map_location=map_location)
    teacher.load_state_dict(payload["teacher"], strict=strict)
    student_f.load_state_dict(payload["student_F"], strict=strict)
    g_noising.load_state_dict(payload["g_noising"], strict=strict)
    teacher.eval()
    student_f.eval()
    g_noising.eval()
    return payload

