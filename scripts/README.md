# Project Scripts

This directory contains project-level entry points. Framework source trees such
as `LLaMA-Factory/`, `EasyR1/`, and `Evaluation/` are kept mostly as upstream
code plus configuration.

- `data/`: data conversion and preprocessing scripts.
- `train/`: runnable training entry points and project-specific training configs.
- `eval/`: runnable evaluation entry points.

Run shell scripts from the repository root, or call them by path. They resolve
the repository root automatically before launching framework code.
