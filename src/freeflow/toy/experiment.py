from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from freeflow.common import count_params, set_seed
from freeflow.io import ensure_dir, save_json

from .config import ToyConfig
from .data import ring_centers, sample_mismatched, sample_true
from .evaluation import evaluate_flowmap_model, evaluate_velocity_model
from .models import FlowMapMLP, NoisingMLP, VelocityMLP
from .plots import save_metric_curves, save_scatter
from .training import (
    train_student_data_based,
    train_student_freeflow,
    train_student_vel_clone,
    train_teacher,
)


def run_toy_experiment(
    config: ToyConfig,
    device,
    output_dir: str | Path,
) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(output_dir)
    centers = ring_centers(config, device)

    teacher = VelocityMLP().to(device)
    student_vel = VelocityMLP().to(device)
    student_f = FlowMapMLP().to(device)
    g_noising = NoisingMLP().to(device)

    teacher = train_teacher(
        teacher,
        config=config,
        device=device,
        x0_sampler=lambda n: sample_true(config, centers, n, device)[0],
    )
    student_vel = train_student_vel_clone(student_vel, teacher, config=config, device=device)
    student_f, g_noising = train_student_freeflow(
        student_f,
        g_noising,
        teacher,
        config=config,
        device=device,
    )

    teacher_eval = evaluate_velocity_model(teacher, config, device, centers, steps=config.ode_steps, n_gen=config.n_eval)
    baseline_eval = evaluate_velocity_model(student_vel, config, device, centers, steps=config.ode_steps, n_gen=config.n_eval)
    freeflow_eval = evaluate_flowmap_model(student_f, config, device, centers, n_gen=config.n_eval)

    mismatch_rows: list[dict[str, float]] = []
    for strength in config.mismatch_levels:
        student = VelocityMLP().to(device)
        student = train_student_data_based(
            student,
            teacher,
            config=config,
            device=device,
            x0_sampler=lambda n, strength=strength: sample_mismatched(config, centers, n, strength, device),
        )
        result = evaluate_velocity_model(student, config, device, centers, steps=config.ode_steps, n_gen=config.n_eval)
        mismatch_rows.append(
            {
                "mismatch_strength": strength,
                "swd": result["swd"],
                "nll": result["nll"],
            }
        )

    nfes = (1, 2, 4, 8, 16, 32)
    teacher_nfe = [evaluate_velocity_model(teacher, config, device, centers, steps=nfe, n_gen=config.n_eval) for nfe in nfes]
    baseline_nfe = [evaluate_velocity_model(student_vel, config, device, centers, steps=nfe, n_gen=config.n_eval) for nfe in nfes]
    freeflow_nfe = evaluate_flowmap_model(student_f, config, device, centers, n_gen=config.n_eval)

    save_scatter(output_dir / "teacher_samples.png", teacher_eval["gen"], centers, "Teacher")
    save_scatter(output_dir / "velocity_clone_samples.png", baseline_eval["gen"], centers, "Velocity-clone student")
    save_scatter(output_dir / "freeflow_samples.png", freeflow_eval["gen"], centers, "FreeFlow student")
    save_metric_curves(output_dir / "quality_vs_nfe.png", nfes, teacher_nfe, baseline_nfe, freeflow_nfe)

    summary = {
        "config": asdict(config),
        "device": str(device),
        "parameter_counts": {
            "teacher": count_params(teacher),
            "velocity_clone": count_params(student_vel),
            "flow_map": count_params(student_f),
            "noising": count_params(g_noising),
        },
        "teacher": {"nll": teacher_eval["nll"], "swd": teacher_eval["swd"]},
        "velocity_clone": {"nll": baseline_eval["nll"], "swd": baseline_eval["swd"]},
        "freeflow": {"nll": freeflow_eval["nll"], "swd": freeflow_eval["swd"]},
        "mismatch_sweep": mismatch_rows,
        "quality_vs_nfe": {
            "teacher": [{"nfe": row["nfe"], "nll": row["nll"], "swd": row["swd"]} for row in teacher_nfe],
            "velocity_clone": [{"nfe": row["nfe"], "nll": row["nll"], "swd": row["swd"]} for row in baseline_nfe],
            "freeflow": {"nfe": freeflow_nfe["nfe"], "nll": freeflow_nfe["nll"], "swd": freeflow_nfe["swd"]},
        },
    }
    save_json(output_dir / "metrics.json", summary)
    return summary

