# FreeFlow Reimplementation

This repository contains a PyTorch implementation inspired by _Flow Map Distillation Without Data_ by Shangyuan Tong et al., developed as part of an M2 Data Science project at Ecole polytechnique.

The project studies how to distill a multi-step generative flow into a one-step flow map without using training data for the student. It includes a toy 2D setting for intuition and quantitative analysis, and a MNIST setting to test the method on image generation with lightweight architectures.

## What is included

- A toy 2D implementation of data-free flow distillation
- A MNIST adaptation with a lightweight U-Net teacher and FreeFlow student
- Training, evaluation, plotting, and checkpoint utilities
- Scripts to reproduce experiments and generate outputs

## Projects' motivations and takeaways

This project sits at the intersection of generative modeling, distillation, and efficient inference:

- It explores how to compress iterative generative dynamics into a much cheaper one-step generator.
- It compares data-free distillation against data-based distillation under dataset mismatch.
- It combines research replication, experimental design, and practical PyTorch implementation.

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

## Method overview

The repository implements three main components:

- A rectified-flow teacher trained to map noise toward data.
- A data-free velocity-cloning baseline student.
- A FreeFlow student that learns a flow map together with an auxiliary noising model, following the core ideas of the paper.

The toy setup is useful for visualizing mode coverage, mismatch sensitivity, and the quality-versus-efficiency trade-off. The MNIST setup extends the same ideas to image generation with convolutional architectures.
