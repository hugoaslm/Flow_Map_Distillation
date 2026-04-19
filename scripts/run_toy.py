from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freeflow.common import get_device
from freeflow.toy.config import ToyConfig
from freeflow.toy.experiment import run_toy_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the toy 2D FreeFlow experiment.")
    parser.add_argument("--output-dir", default="outputs/toy")
    parser.add_argument("--teacher-steps", type=int, default=2500)
    parser.add_argument("--student-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = ToyConfig(
        teacher_steps=args.teacher_steps,
        student_steps=args.student_steps,
        batch_size=args.batch_size,
    )
    summary = run_toy_experiment(config, get_device(args.device), args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
