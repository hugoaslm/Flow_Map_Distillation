from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MnistConfig:
    seed: int = 0
    img_size: int = 28
    channels: int = 1
    num_workers: int = 2
    teacher_steps: int = 5000
    student_steps: int = 5000
    batch_size: int = 128
    lr_teacher: float = 2e-4
    lr_student: float = 2e-4
    grad_clip: float = 1.0
    ode_steps: int = 60
    t_min: float = 1e-3
    n_eval: int = 2048

