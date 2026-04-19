# FreeFlow Reimplementation

This repository is a cleaned-up version of an M2 Data Science course project at Ecole polytechnique on the paper _Flow Map Distillation Without Data_ by Shangyuan Tong et al.

The original class submission was developed in a single dense notebook. This repo turns that work into a more standard Python project with reusable modules, runnable experiment entry points, and a clearer separation between models, training loops, evaluation, and presentation code.

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

## Notes on scope

- The main showcase code focuses on the toy and MNIST experiments because those were the parts we were able to run and analyze credibly with available compute.
- The CIFAR-10 / DiT attempt from the notebook is intentionally not exposed as a primary script. It was exploratory, did not converge reliably, and would weaken the presentation if treated as production code.
- Training defaults are kept close to the notebook, but CLI arguments make it easy to shorten runs for quick checks.

## Original material

- Original notebook: [freeflow_experiments_dana_garcia_lefebvre_anselme.ipynb](/C:/Users/hugo2/OneDrive/Documents/Codex/freeflow_experiments_dana_garcia_lefebvre_anselme.ipynb)

