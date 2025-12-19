# Mean-Field Reinforcement Learning: TD3 vs SAC

This repository accompanies a Master's thesis on mean-field control problems
and compares TD3 and SAC in a mean-field MDP with common noise.

## Overview
- Mean-field environment with common noise
- TD3 and SAC implementations in a unified setting
- Evaluation via terminal error and discounted cost
- Focus on conceptual clarity rather than MLOps infrastructure

## Repository structure
- `mf_triangle_modell.py` – Mean-field environment
- `mf_td3.py` – TD3 implementation
- `mf_sac.py` – SAC implementation
- `mf_compare_main.py` – Main experiment script (used for thesis figures)
- `mf_eval.py` – Evaluation routines
- `mf_plotting.py` – Plotting utilities
- `mf_analyze_training.py` – Training diagnostics

## How to run
```bash
python mf_compare_main.py
