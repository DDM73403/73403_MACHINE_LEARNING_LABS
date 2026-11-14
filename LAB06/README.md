# LAB06 - Training Pipelines and Project Proposal

Student: Daisy Mwale
Student ID: 73403

## Overview

This laboratory focuses on building a reproducible machine-learning training pipeline using YAML configuration files, MLflow experiment tracking, PyTorch profiling, performance studies, checkpoint recovery, and deployment-aware project planning.

Dataset used:
- FashionMNIST

Model used:
- AttentionModel

## Project Structure

LAB06
├── checkpoints/
│   └── model.pth
│
├── results/
│
├── src/
│   ├── train.py
│   ├── utils.py
│   ├── config.yaml
│   ├── torch_profile.py
│   ├── experiments.py
│   ├── batch_size_study.py
│   ├── scheduler_study.py
│   ├── gradient_accumulation.py
│   └── checkpoint_recovery.py
│
├── README.md
└── report.md

## Tasks Completed

Task 1:
- YAML-driven training pipeline
- Reproducibility using fixed seed
- MLflow experiment tracking
- Checkpoint saving
- Parameter logging
- Metric logging

Task 2:
- PyTorch profiling of one training epoch

Task 3:
- num_workers performance study

Task 4:
- Batch-size performance study

Task 5:
- Learning-rate scheduler and optimizer comparison

Task 6:
- Gradient accumulation experiment

Task 7:
- Checkpoint recovery and resume testing

Task 8:
- Individual project proposal

## Main Results

Training Accuracy:
- 83.40%

Best Batch Size:
- 512

Best Throughput:
- 4767.38 samples/s

Best Optimizer and Scheduler:
- Adam + StepLR (84.64%)

## Technologies Used

- Python 3.11
- PyTorch
- TorchVision
- MLflow
- PyYAML
- NumPy

## Author

Daisy Mwale
73403