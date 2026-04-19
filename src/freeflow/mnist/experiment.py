from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from freeflow.common import count_params, set_seed
from freeflow.io import ensure_dir, save_json

from .checkpoints import save_checkpoint
from .config import MnistConfig
from .data import MnistDataModule
from .evaluation import evaluate_models_vs_nfe
from .models import UNetLite
from .plots import save_sample_grid, save_swd_curve
from .training import euler_sample_images, sample_one_step_freeflow, train_student_freeflow, train_teacher


def run_mnist_experiment(
    config: MnistConfig,
    device,
    output_dir: str | Path,
    data_root: str = "./data",
) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(output_dir)
    data = MnistDataModule(config, device=device, root=data_root)

    teacher = UNetLite(in_ch=config.channels).to(device)
    student_f = UNetLite(in_ch=config.channels).to(device)
    g_noising = UNetLite(in_ch=config.channels).to(device)

    teacher = train_teacher(teacher, config=config, data=data)
    student_f, g_noising = train_student_freeflow(
        student_f,
        g_noising,
        teacher,
        config=config,
        device=device,
    )

    teacher_samples = euler_sample_images(teacher, config, 64, steps=32, device=device)
    freeflow_samples = sample_one_step_freeflow(student_f, config, 64, device=device)
    rows, teacher_sweeps, freeflow_eval_samples = evaluate_models_vs_nfe(
        teacher,
        student_f,
        config=config,
        data=data,
        device=device,
    )

    save_sample_grid(output_dir / "teacher_samples.png", teacher_samples, "Teacher samples (Euler 32 steps)")
    save_sample_grid(output_dir / "freeflow_samples.png", freeflow_samples, "FreeFlow samples (1 step)")
    save_sample_grid(output_dir / "freeflow_eval_samples.png", freeflow_eval_samples[:64], "FreeFlow evaluation samples")
    save_sample_grid(output_dir / "teacher_eval_samples.png", teacher_sweeps[32][:64], "Teacher evaluation samples")
    save_swd_curve(output_dir / "quality_vs_nfe.png", rows, (1, 2, 4, 8, 16, 32))
    save_checkpoint(output_dir / "mnist_freeflow_ckpt.pt", teacher, student_f, g_noising, config=config)

    summary = {
        "config": asdict(config),
        "device": str(device),
        "parameter_counts": {
            "teacher": count_params(teacher),
            "flow_map": count_params(student_f),
            "noising": count_params(g_noising),
        },
        "quality_vs_nfe": rows,
    }
    save_json(output_dir / "metrics.json", summary)
    return summary

