from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToyConfig:
    seed: int = 0
    k: int = 8
    ring_radius: float = 2.5
    data_sigma: float = 0.15
    batch_size: int = 512
    teacher_steps: int = 2500
    student_steps: int = 2000
    lr_teacher: float = 2e-3
    lr_student: float = 2e-3
    grad_clip: float = 1.0
    ode_steps: int = 60
    traj_steps: int = 40
    t_min: float = 1e-3
    mismatch_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    n_eval: int = 8000
    n_plot: int = 6000

