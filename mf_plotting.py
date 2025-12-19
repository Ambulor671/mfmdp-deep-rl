# -*- coding: utf-8 -*-
"""
Plotting utilities for TD3 vs SAC comparisons.
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np


def load_curve(csv_path: str):
    if not os.path.isfile(csv_path):
        return [], [], [], [], [], []
    eps, j_mean, j_std, e_mean, e_std, n_seeds = [], [], [], [], [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            eps.append(int(row["episode"]))
            j_mean.append(float(row["Jeval_mean"]))
            j_std.append(float(row["Jeval_std"]))
            e_mean.append(float(row["end_mean"]))
            e_std.append(float(row["end_std"]))
            n_seeds.append(int(row.get("seed_n", 1)))
    return (
        np.asarray(eps, dtype=int),
        np.asarray(j_mean, dtype=float),
        np.asarray(j_std, dtype=float),
        np.asarray(e_mean, dtype=float),
        np.asarray(e_std, dtype=float),
        np.asarray(n_seeds, dtype=int),
    )


def smooth(x, k: int = 1) -> np.ndarray:
    if k <= 1 or len(x) < 3:
        return np.array(x, dtype=float)
    k = min(int(k) | 1, len(x) if len(x) % 2 == 1 else len(x) - 1)
    pad = k // 2
    arr = np.pad(np.array(x, dtype=float), (pad, pad), mode="edge")
    kern = np.ones(k) / k
    return np.convolve(arr, kern, mode="valid")


def align_on_common_eps(eps_a: np.ndarray, vals_a: np.ndarray,
                        eps_b: np.ndarray, vals_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = sorted(set(eps_a.tolist()) & set(eps_b.tolist()))
    xs = np.asarray(common, dtype=int)
    if len(xs) == 0:
        return xs, np.array([]), np.array([])

    idx_a = {int(e): i for i, e in enumerate(eps_a)}
    idx_b = {int(e): i for i, e in enumerate(eps_b)}
    va = np.array([vals_a[idx_a[int(e)]] for e in xs], dtype=float)
    vb = np.array([vals_b[idx_b[int(e)]] for e in xs], dtype=float)
    return xs, va, vb


def band(std: np.ndarray, n: np.ndarray, use_sem: bool = True) -> np.ndarray:
    if not use_sem:
        return std
    n_safe = np.maximum(n.astype(float), 1.0)
    return std / np.sqrt(n_safe)


def boxplot(values_list, labels, ylabel, outfile, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.boxplot(values_list, labels=labels, showmeans=True)
    if title:
        plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.savefig(outfile, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_heatmaps(trajs: List[np.ndarray], algo: str, outdir: str, what: str, seeds_list: List[int]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for idx, arr in enumerate(trajs):
        fig = plt.figure()
        if what == "mu":
            plt.imshow(arr.T, aspect="auto", origin="lower", vmax=0.3)
            plt.colorbar(label=r"$\mu$ value")
            plt.ylabel("state s")
        else:
            plt.imshow(arr.T, aspect="auto", origin="lower", vmin=-0.7, vmax=0.5)
            plt.colorbar(label="action")
            plt.ylabel("action dim")
        plt.xlabel("t")
        seed_label = seeds_list[idx] if idx < len(seeds_list) else idx
        fname = f"{algo.lower()}_{what}_over_time_seed{seed_label}.png"
        fig.savefig(os.path.join(outdir, fname), bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_training_curves(td3_curve_csv: str, sac_curve_csv: str, outdir: str, smooth_k: int = 3, use_sem: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    td3_eps, td3_Jm, td3_Js, td3_Em, td3_Es, td3_N = load_curve(td3_curve_csv)
    sac_eps, sac_Jm, sac_Js, sac_Em, sac_Es, sac_N = load_curve(sac_curve_csv)

    xs = sorted(set(td3_eps.tolist()) & set(sac_eps.tolist()))
    if len(xs) == 0:
        print("Note: No overlapping evaluation checkpoints (align eval_every).")
        return

    xs = np.asarray(xs, dtype=int)

    def _align(eps, vals):
        idx = {int(e): i for i, e in enumerate(eps)}
        return np.array([vals[idx[int(e)]] for e in xs], dtype=float)

    td3Jm = smooth(_align(td3_eps, td3_Jm), k=smooth_k)
    sacJm = smooth(_align(sac_eps, sac_Jm), k=smooth_k)
    td3Js = smooth(_align(td3_eps, band(td3_Js, td3_N, use_sem=use_sem)), k=smooth_k)
    sacJs = smooth(_align(sac_eps, band(sac_Js, sac_N, use_sem=use_sem)), k=smooth_k)

    fig = plt.figure()
    plt.plot(xs, td3Jm, label=r"TD3 – $\hat J_{eval}$")
    plt.fill_between(xs, td3Jm - td3Js, td3Jm + td3Js, alpha=0.25, label="TD3 band")
    plt.plot(xs, sacJm, label=r"SAC – $\hat J_{eval}$")
    plt.fill_between(xs, sacJm - sacJs, sacJm + sacJs, alpha=0.25, label="SAC band")
    plt.xlabel("episode (eval checkpoints)")
    plt.ylabel(r"$\hat J_{\mathrm{eval}}$")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    fig.savefig(os.path.join(outdir, "train_curve_Jeval_td3_vs_sac_bands.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    td3Em = smooth(_align(td3_eps, td3_Em), k=smooth_k)
    sacEm = smooth(_align(sac_eps, sac_Em), k=smooth_k)
    td3Es = smooth(_align(td3_eps, band(td3_Es, td3_N, use_sem=use_sem)), k=smooth_k)
    sacEs = smooth(_align(sac_eps, band(sac_Es, sac_N, use_sem=use_sem)), k=smooth_k)

    fig = plt.figure()
    plt.plot(xs, td3Em, label=r"TD3 – $\|\mu_T-\mu_{target}\|_2^2$")
    plt.fill_between(xs, td3Em - td3Es, td3Em + td3Es, alpha=0.25, label="TD3 band")
    plt.plot(xs, sacEm, label=r"SAC – $\|\mu_T-\mu_{target}\|_2^2$")
    plt.fill_between(xs, sacEm - sacEs, sacEm + sacEs, alpha=0.25, label="SAC band")
    plt.xlabel("episode (eval checkpoints)")
    plt.ylabel(r"$\|\mu_T-\mu_{target}\|_2^2$")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    fig.savefig(os.path.join(outdir, "train_curve_Endfehler_td3_vs_sac_bands.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
