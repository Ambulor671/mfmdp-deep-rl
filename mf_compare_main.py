# -*- coding: utf-8 -*-
"""
mf_compare_main.py

Train TD3 and SAC on MFCommonNoiseTriangleModel and compare:
- terminal error ||mu_T - mu_target||^2
- discounted evaluation cost J_eval
- training curves from intermittent evaluation
- boxplots and per-seed heatmaps

This script orchestrates training, evaluation, and plotting.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import time
from typing import List

import numpy as np

#Local Imports

from mf_triangle_modell import MFCommonNoiseTriangleModel

from mf_io import ensure_dir
from mf_td3 import TD3_Config, train_td3
from mf_sac import SAC_Config, train_sac
from mf_eval import eval_end_l2
from mf_plotting import (
    boxplot,
    plot_heatmaps,
    plot_training_curves,
)
from mf_analyze_training import analyze_run




#Configuration

MODEL_PARAMS = dict(
    gamma=0.99,
    p_jump=0.05,
    sigma=0.2,
    id_p_jump=0.0,
    id_sigma=0.0,
    m=11,
    seed=7,
)

T = 100
GAMMA_EVAL = MODEL_PARAMS["gamma"]

EVAL_SEEDS_CURVE: List[int] = list(range(5))
FINAL_EVAL_SEEDS: List[int] = list(range(10))


#Main

def main() -> None:
    # TD3
    td3_cfg = TD3_Config(
        episodes=300,
        T=T,
        device="cpu",
        action_low=-1.0,
        action_high=1.0,
        out_dir="runs/mf_td3",
        run_name="td3_triangle",
        eval_every=10,
        eval_episodes=3,
        policy_noise_sigma=0.2,
        noise_clip=0.2,
        policy_delay=2,
        explore_sigma=0.05,
        lr_actor=1e-4,
        lr_critic=1e-4,
    )

    print("[TD3] Training starts ...")
    t0 = time.time()
    td3_agent, td3_run = train_td3(
        MFCommonNoiseTriangleModel(**MODEL_PARAMS),
        td3_cfg,
        eval_fn=eval_end_l2,
        eval_seeds=EVAL_SEEDS_CURVE,
        eval_model_params=MODEL_PARAMS,
        eval_T=T,
        eval_gamma=GAMMA_EVAL,
        on_eval=lambda ep, row: print(
            f"[TD3 eval] ep={ep} J={row['Jeval_mean']:.3f} End={row['end_mean']:.3f}"
        ),
    )
    print(f"[TD3] done in {(time.time()-t0)/60:.2f} min, artifacts: {td3_run}")

    # SAC
    sac_cfg = SAC_Config(
        episodes=300,
        T=T,
        device="cpu",
        action_low=-1.0,
        action_high=1.0,
        out_dir="runs/mf_sac",
        run_name="sac_triangle",
        target_entropy=-MODEL_PARAMS["m"],
        weight_logp_by_mu=True,
    )

    print("[SAC] Training starts ...")
    t0 = time.time()
    sac_agent, sac_run = train_sac(
        MFCommonNoiseTriangleModel(**MODEL_PARAMS),
        sac_cfg,
        eval_fn=eval_end_l2,
        eval_seeds=EVAL_SEEDS_CURVE,
        eval_model_params=MODEL_PARAMS,
        eval_T=T,
        eval_gamma=GAMMA_EVAL,
        on_eval=lambda ep, row: print(
            f"[SAC eval] ep={ep} J={row['Jeval_mean']:.3f} End={row['end_mean']:.3f}"
        ),
    )
    print(f"[SAC] done in {(time.time()-t0)/60:.2f} min, artifacts: {sac_run}")

    # Policies

    def td3_policy(mu: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return td3_agent.act(mu, explore=not deterministic)

    def sac_policy(mu: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return sac_agent.act(mu, explore=not deterministic)

    # Final Evaluation 
    print("\n=== EVALUATION ===")

    td3_vals, td3_Jeval, td3_trajs, td3_acts = eval_end_l2(
        MODEL_PARAMS, T, td3_policy, True, FINAL_EVAL_SEEDS,
        gamma=GAMMA_EVAL, return_traj=True
    )
    sac_vals, sac_Jeval, sac_trajs, sac_acts = eval_end_l2(
        MODEL_PARAMS, T, sac_policy, True, FINAL_EVAL_SEEDS,
        gamma=GAMMA_EVAL, return_traj=True
    )

    print(
        f"Terminal error  TD3: {td3_vals.mean():.4f} ± {td3_vals.std(ddof=1):.4f} | "
        f"SAC: {sac_vals.mean():.4f} ± {sac_vals.std(ddof=1):.4f}"
    )
    print(
        f"J_eval         TD3: {td3_Jeval.mean():.4f} ± {td3_Jeval.std(ddof=1):.4f} | "
        f"SAC: {sac_Jeval.mean():.4f} ± {sac_Jeval.std(ddof=1):.4f}"
    )

    #Plotting
    
    plot_root = os.path.join("runs", "plots", str(int(time.time())))
    ensure_dir(plot_root)

    heat_mu_dir = os.path.join(plot_root, "heatmaps_mu")
    heat_act_dir = os.path.join(plot_root, "heatmaps_actions")
    ensure_dir(heat_mu_dir)
    ensure_dir(heat_act_dir)

    boxplot(
        [td3_vals, sac_vals],
        ["TD3", "SAC"],
        ylabel=r"$\|\mu_T - \mu_{\mathrm{target}}\|_2^2$",
        outfile=os.path.join(plot_root, "box_end_l2.png"),
        title=None,
    )

    boxplot(
        [td3_Jeval, sac_Jeval],
        ["TD3", "SAC"],
        ylabel=r"$\hat J_{\mathrm{eval}}$",
        outfile=os.path.join(plot_root, "box_Jeval.png"),
        title=None,
    )

    plot_heatmaps(td3_trajs, "TD3", heat_mu_dir, "mu", FINAL_EVAL_SEEDS)
    plot_heatmaps(sac_trajs, "SAC", heat_mu_dir, "mu", FINAL_EVAL_SEEDS)
    plot_heatmaps(td3_acts,  "TD3", heat_act_dir, "act", FINAL_EVAL_SEEDS)
    plot_heatmaps(sac_acts,  "SAC", heat_act_dir, "act", FINAL_EVAL_SEEDS)

    curve_dir = os.path.join(plot_root, "curves")
    ensure_dir(curve_dir)

    plot_training_curves(
        os.path.join(td3_run, "eval_curve.csv"),
        os.path.join(sac_run, "eval_curve.csv"),
        curve_dir,
        smooth_k=3,
        use_sem=True,
    )


    # Analysis of Training 
    analyze_run(td3_run, out_subdir="analysis")
    analyze_run(sac_run, out_subdir="analysis")

    print("All results written to:", os.path.abspath(plot_root))


if __name__ == "__main__":
    main()

