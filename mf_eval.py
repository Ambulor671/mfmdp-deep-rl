# -*- coding: utf-8 -*-
"""
Evaluation utilities for MFCommonNoiseTriangleModel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mf_triangle_modell import MFCommonNoiseTriangleModel


@torch.no_grad()
def eval_end_l2(
    model_params: Dict[str, Any],
    T: int,
    policy_call,
    deterministic: bool,
    seeds: List[int],
    gamma: float = 1.0,
    return_traj: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    end_vals: List[float] = []
    disc_costs: List[float] = []

    mu_trajs: Optional[List[np.ndarray]] = [] if return_traj else None
    act_trajs: Optional[List[np.ndarray]] = [] if return_traj else None

    for s in seeds:
        env = MFCommonNoiseTriangleModel(**model_params)
        mu = env.reset(seed=s)

        if return_traj:
            traj_mu = [mu.copy()]
            traj_a: List[np.ndarray] = []

        total_cost = 0.0
        disc = 1.0

        for _t in range(T):
            a = policy_call(mu, deterministic=deterministic)
            mu, r, _info = env.step(mu, a, train=False)

            total_cost += disc * (-float(r))
            disc *= gamma

            if return_traj:
                traj_mu.append(mu.copy())
                traj_a.append(np.asarray(a, dtype=float).copy())

        diff = mu.astype(np.float64) - env.mu_target
        end_vals.append(float(np.sum(diff * diff)))
        disc_costs.append(float(total_cost))

        if return_traj and mu_trajs is not None and act_trajs is not None:
            mu_trajs.append(np.asarray(traj_mu, dtype=float))
            act_trajs.append(np.asarray(traj_a, dtype=float))

    return (
        np.asarray(end_vals, dtype=float),
        np.asarray(disc_costs, dtype=float),
        mu_trajs,
        act_trajs,
    )
