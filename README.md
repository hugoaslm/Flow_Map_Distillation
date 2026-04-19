# FreeFlow Reimplementation

This repository is a cleaned-up version of an M2 Data Science course project at Ecole polytechnique on the paper _Flow Map Distillation Without Data_ by Shangyuan Tong et al.

## What is included

- A toy 2D reproduction of the paper's data-free distillation setup
- A MNIST adaptation using a lightweight U-Net teacher and FreeFlow student
- Evaluation helpers for sample quality and efficiency trade-offs
- Plot generation and checkpoint utilities
- The original notebook kept as a historical reference

## Project layout

```text
src/freeflow/
  common.py
  io.py
  toy/
    config.py
    data.py
    models.py
    training.py
    evaluation.py
    plots.py
    experiment.py
  mnist/
    config.py
    data.py
    models.py
    training.py
    evaluation.py
    checkpoints.py
    plots.py
    experiment.py
scripts/
  run_toy.py
  run_mnist.py
```

## Quick start

Create a virtual environment, install the package in editable mode, then run one of the experiments:

```bash
pip install -e .
python scripts/run_toy.py --output-dir outputs/toy
python scripts/run_mnist.py --output-dir outputs/mnist
```
