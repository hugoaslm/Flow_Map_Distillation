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
from freeflow.mnist.config import MnistConfig
from freeflow.mnist.experiment import run_mnist_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MNIST FreeFlow experiment.")
    parser.add_argument("--output-dir", default="outputs/mnist")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--teacher-steps", type=int, default=5000)
    parser.add_argument("--student-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = MnistConfig(
        teacher_steps=args.teacher_steps,
        student_steps=args.student_steps,
        batch_size=args.batch_size,
    )
    summary = run_mnist_experiment(
        config=config,
        device=get_device(args.device),
        output_dir=args.output_dir,
        data_root=args.data_root,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
