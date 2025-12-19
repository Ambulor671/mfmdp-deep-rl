# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Optional, Sequence, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or k <= 1:
        return x
    k = int(k)
    k = min(k, x.size)
    if k % 2 == 0:
        k = max(1, k - 1)
    pad = k // 2
    kernel = np.ones(k, dtype=float) / k
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, kernel, mode="valid")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_series(
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    out_path: str,
    smooth: int = 1,
    vlines: Optional[List[int]] = None,
    ylog: bool = False,
) -> None:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return

    y_s = moving_average(y, smooth)
    n = min(len(y), len(y_s))
    if n == 0:
        return

    y_s = y_s[:n]
    x = np.arange(n)

    fig = plt.figure()
    plt.plot(x, y_s)

    if ylog:
        plt.yscale("log")

    if vlines:
        for vx in vlines:
            if 0 <= vx < n:
                plt.axvline(vx, linestyle="--", linewidth=0.8, alpha=0.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_histograms(arr: np.ndarray, times: Sequence[int], out_path: str) -> None:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array (time, dim) or (T+1, dim) for mu/actions")

    T = arr.shape[0]
    times = [t for t in times if 0 <= t < T]
    if not times:
        return

    cols = len(times)
    fig = plt.figure(figsize=(4 * cols, 3.5))
    for i, t in enumerate(times, 1):
        ax = fig.add_subplot(1, cols, i)
        ax.hist(arr[t], bins=30)
        ax.set_xlabel("Wert")
        ax.set_ylabel("Häufigkeit")
        ax.text(0.02, 0.98, f"t={t}", transform=ax.transAxes, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_times_arg(times_arg: str, T_val: int) -> List[int]:
    raw = [tok.strip() for tok in times_arg.split(",") if tok.strip()]
    out: List[int] = []
    for tok in raw:
        if tok == "0":
            out.append(0)
        elif tok.lower() == "mid":
            out.append(max(0, T_val // 2))
        elif tok.lower() == "last":
            out.append(max(0, T_val - 1))
        else:
            try:
                out.append(int(tok))
            except ValueError:
                pass
    out = sorted(set([t for t in out if 0 <= t < max(1, T_val)]))
    if not out and T_val > 0:
        out = [0, max(0, T_val // 2), max(0, T_val - 1)]
        out = sorted(set(out))
    return out


def is_sac(df_steps: pd.DataFrame) -> bool:
    sac_cols = {"alpha", "logp_pi", "q_pi_mean", "mean_std", "sum_log_std"}
    return any(col in df_steps.columns for col in sac_cols)


def _rolling_mean_std(y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size == 0:
        return y, np.zeros_like(y)

    window = min(int(window), y.size if y.size > 0 else 1)
    if window % 2 == 0:
        window = max(1, window - 1)

    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window

    mu = np.convolve(ypad, kernel, mode="valid")
    mu2 = np.convolve(ypad**2, kernel, mode="valid")
    var = np.maximum(0.0, mu2 - mu**2)
    std = np.sqrt(var)
    return mu, std


def find_convergence_index(
    y: np.ndarray,
    *,
    window: int = 20,
    mean_tol: float = 1e-3,
    std_tol: float = 1e-2,
    min_episodes: int = 40
) -> Dict[str, Optional[float]]:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return {"conv_episode": None, "conv_mean": None, "conv_std": None}

    mu, sd = _rolling_mean_std(y, window)
    mu_shift = np.concatenate([np.full(window, np.nan), mu[:-window]]) if n > window else np.full(n, np.nan)

    for t in range(max(min_episodes, window), n):
        if np.isnan(mu_shift[t]):
            continue
        scale = max(1.0, abs(mu[t]))
        mean_ok = abs(mu[t] - mu_shift[t]) <= mean_tol * scale
        std_ok = sd[t] <= std_tol * scale
        if mean_ok and std_ok:
            return {"conv_episode": int(t), "conv_mean": float(mu[t]), "conv_std": float(sd[t])}

    return {"conv_episode": None, "conv_mean": None, "conv_std": None}


def summarize_final_window(y: np.ndarray, window: int = 20) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return float("nan"), float("nan")
    w = max(1, min(int(window), y.size))
    tail = y[-w:]
    return float(np.mean(tail)), float(np.std(tail))


def write_convergence_txt(
    out_dir: str,
    *,
    conv_basis: str,
    conv_info: Dict[str, Optional[float]],
    ep_metrics: Dict[str, np.ndarray],
    eval_metrics: Dict[str, np.ndarray],
    conv_window: int,
    mean_tol: float,
    std_tol: float
) -> str:
    lines: List[str] = []
    lines.append("Convergence report")
    lines.append("==================")
    lines.append(f"basis_series: {conv_basis}")
    lines.append(f"criteria: window={conv_window}, mean_tol={mean_tol}, std_tol={std_tol}")
    lines.append("")

    tstar = conv_info.get("conv_episode")
    mu = conv_info.get("conv_mean")
    sd = conv_info.get("conv_std")

    lines.append("Convergence (basis):")
    lines.append(f"  episode*: {tstar if tstar is not None else 'None'}")
    lines.append(f"  mean@*:  {mu if mu is not None else 'None'}")
    lines.append(f"  std@*:   {sd if sd is not None else 'None'}")
    lines.append("")
    if tstar is None:
        lines.append("NOTE: convergence not detected with given criteria.")
        lines.append("")

    def _append_block(title: str, series: Dict[str, np.ndarray]) -> None:
        if not series:
            return
        lines.append(title)
        for k, arr in series.items():
            m, s = summarize_final_window(arr, window=conv_window)
            lines.append(f"  {k:30s} mean_last{conv_window:>2d}: {m:.6g}   std_last{conv_window:>2d}: {s:.6g}")
        lines.append("")

    _append_block("Episode metrics (final window):", ep_metrics)
    _append_block("Eval metrics (final window):", eval_metrics)

    out_path = os.path.join(out_dir, "convergence.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def analyze_run(
    run_dir: str,
    out_subdir: str = "analysis",
    smooth: int = 11,
    hist_times: str = "0,mid,last",
    conv_series: Optional[str] = None,
    conv_window: int = 10,
    conv_mean_tol: float = 5e-3,
    conv_std_tol: float = 5e-2
) -> str:
    out_dir = os.path.join(run_dir, out_subdir)
    ensure_dir(out_dir)

    cfg_path = os.path.join(run_dir, "config.json")
    T_config: Optional[int] = None
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        T_config = int(cfg.get("T", 0) or 0) or None

    steps_path = os.path.join(run_dir, "train_steps.csv")
    df_steps = None
    if os.path.exists(steps_path):
        df_steps = pd.read_csv(steps_path).sort_values(["episode", "t"]).reset_index(drop=True)
        vlines = list(range(T_config, len(df_steps), T_config)) if T_config else None

        base_specs = [
            # col,           fname,                  xlabel,        ylabel,                       ylog
            ("reward",       "steps_reward.png",      "Zeitschritt", "Belohnung",                  False),
            ("stage_cost",   "steps_stage_cost.png",  "Zeitschritt", "Stufenkosten",               False),
            ("mu_delta_sq",  "steps_mu_delta_sq.png", "Zeitschritt", r"$\|\mu_t-\mu_{t+1}\|^2$",   False),
            ("actor_loss",   "steps_actor_loss.png",  "Zeitschritt", "Actor-Loss",                 False),
            ("critic_loss",  "steps_critic_loss.png", "Zeitschritt", "Critic-Loss",                True),
        ]
        for col, fname, xlabel, ylabel, ylog in base_specs:
            if col in df_steps.columns:
                plot_series(
                    df_steps[col].to_numpy(dtype=float),
                    xlabel=xlabel,
                    ylabel=ylabel,
                    out_path=os.path.join(out_dir, fname),
                    smooth=smooth,
                    vlines=vlines,
                    ylog=ylog,
                )

        if is_sac(df_steps):
            sac_specs = [
                ("alpha",       "steps_alpha.png",       "Zeitschritt", "α"),
                ("logp_pi",     "steps_logp_pi.png",     "Zeitschritt", "log π"),
                ("q_pi_mean",   "steps_q_pi_mean.png",   "Zeitschritt", "Q"),
                ("mean_std",    "steps_mean_std.png",    "Zeitschritt", r"$E_{\mu}[\sigma_{(i)}]$"),
                ("sum_log_std", "steps_sum_log_std.png", "Zeitschritt", "Σ log σ"),
            ]
            for col, fname, xlabel, ylabel in sac_specs:
                if col in df_steps.columns:
                    plot_series(
                        df_steps[col].to_numpy(dtype=float),
                        xlabel=xlabel,
                        ylabel=ylabel,
                        out_path=os.path.join(out_dir, fname),
                        smooth=max(1, min(smooth, 7)),
                        vlines=vlines,
                        ylog=False,
                    )

    ep_path = os.path.join(run_dir, "train_episodes.csv")
    df_ep = None
    if os.path.exists(ep_path):
        df_ep = pd.read_csv(ep_path).sort_values("episode")
        ep_specs = [
            ("sum_stage_cost",  "episodes_sum_stage_cost.png",  "Episode", "Summe Stufenkosten"),
            ("avg_stage_cost",  "episodes_avg_stage_cost.png",  "Episode", "Mittlere Stufenkosten"),
            ("discounted_cost", "episodes_discounted_cost.png", "Episode", "Diskontierte Kosten pro Episode"),
            ("return_reward",   "episodes_return_reward.png",   "Episode", "Return"),
        ]
        for col, fname, xlabel, ylabel in ep_specs:
            if col in df_ep.columns:
                plot_series(
                    df_ep[col].to_numpy(dtype=float),
                    xlabel=xlabel,
                    ylabel=ylabel,
                    out_path=os.path.join(out_dir, fname),
                    smooth=max(1, min(smooth, 7)),
                    vlines=None,
                    ylog=False,
                )

    eval_path = os.path.join(run_dir, "eval.csv")
    df_ev = None
    if os.path.exists(eval_path):
        df_ev = pd.read_csv(eval_path).sort_values("episode")
        ev_specs = [
            ("mean_J_return",        "eval_mean_return.png",          "Episode", "Mittlerer Return"),
            ("mean_discounted_cost", "eval_mean_discounted_cost.png", "Episode", "Mittlere diskontierte Kosten"),
        ]
        for col, fname, xlabel, ylabel in ev_specs:
            if col in df_ev.columns:
                plot_series(
                    df_ev[col].to_numpy(dtype=float),
                    xlabel=xlabel,
                    ylabel=ylabel,
                    out_path=os.path.join(out_dir, fname),
                    smooth=max(1, min(smooth, 5)),
                    vlines=None,
                    ylog=False,
                )

    mu_path = os.path.join(run_dir, "mu_traj.npy")
    a_path = os.path.join(run_dir, "action_traj.npy")
    if os.path.exists(mu_path):
        mu = np.load(mu_path)
        times = parse_times_arg(hist_times, mu.shape[0])
        plot_histograms(mu, times, out_path=os.path.join(out_dir, "mu_histograms.png"))
    if os.path.exists(a_path):
        act = np.load(a_path)
        times = parse_times_arg(hist_times, act.shape[0])
        plot_histograms(act, times, out_path=os.path.join(out_dir, "action_histograms.png"))

    summary: Dict[str, float] = {}

    if df_ep is not None and not df_ep.empty:
        if "discounted_cost" in df_ep.columns:
            summary["best_discounted_cost"] = float(df_ep["discounted_cost"].min())
        if "return_reward" in df_ep.columns:
            summary["best_return_reward"] = float(df_ep["return_reward"].max())

    if df_ev is not None and not df_ev.empty:
        if "mean_discounted_cost" in df_ev.columns:
            summary["best_eval_mean_discounted_cost"] = float(df_ev["mean_discounted_cost"].min())
        if "mean_J_return" in df_ev.columns:
            summary["best_eval_mean_return"] = float(df_ev["mean_J_return"].max())

    conv_basis = "(none)"
    conv_info = {"conv_episode": None, "conv_mean": None, "conv_std": None}
    ep_series_for_report: Dict[str, np.ndarray] = {}
    eval_series_for_report: Dict[str, np.ndarray] = {}

    if df_ep is not None and not df_ep.empty:
        for key in ["return_reward", "discounted_cost", "avg_stage_cost", "sum_stage_cost"]:
            if key in df_ep.columns:
                ep_series_for_report[key] = df_ep[key].to_numpy(dtype=float)

        if conv_series and conv_series in df_ep.columns:
            y = df_ep[conv_series].to_numpy(dtype=float)
            y_use = y if conv_series == "return_reward" else (-y if conv_series == "discounted_cost" else y)
            conv_basis = conv_series if conv_series != "discounted_cost" else "discounted_cost (negated)"
        else:
            y_use = None
            if "return_reward" in df_ep.columns:
                y_use = df_ep["return_reward"].to_numpy(dtype=float)
                conv_basis = "return_reward"
            elif "discounted_cost" in df_ep.columns:
                y_use = -df_ep["discounted_cost"].to_numpy(dtype=float)
                conv_basis = "discounted_cost (negated)"

        if y_use is not None:
            conv_info = find_convergence_index(
                y_use,
                window=conv_window,
                mean_tol=conv_mean_tol,
                std_tol=conv_std_tol,
                min_episodes=max(40, conv_window),
            )
            if conv_info.get("conv_episode") is not None:
                summary["episodes_to_convergence"] = int(conv_info["conv_episode"])

    if df_ev is not None and not df_ev.empty:
        for key in ["mean_J_return", "mean_discounted_cost"]:
            if key in df_ev.columns:
                eval_series_for_report[key] = df_ev[key].to_numpy(dtype=float)

    conv_txt_path = write_convergence_txt(
        out_dir,
        conv_basis=conv_basis,
        conv_info=conv_info,
        ep_metrics=ep_series_for_report,
        eval_metrics=eval_series_for_report,
        conv_window=conv_window,
        mean_tol=conv_mean_tol,
        std_tol=conv_std_tol,
    )
    summary["convergence_report"] = os.path.basename(conv_txt_path)

    if summary:
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Saved analysis outputs to: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a single run directory")
    ap.add_argument("--out_subdir", default="analysis", help="Subdirectory for plots")
    ap.add_argument("--smooth", type=int, default=11,
                    help="Centered moving average window (odd). Use 1 to disable.")
    ap.add_argument("--hist_times", type=str, default="0,mid,last",
                    help="Comma list of times for histograms: indices or keywords 0,mid,last")
    ap.add_argument("--conv_series", type=str, default=None,
                    help="Episode series for convergence, e.g. return_reward or discounted_cost")
    ap.add_argument("--conv_window", type=int, default=10, help="Window for convergence detection")
    ap.add_argument("--conv_mean_tol", type=float, default=5e-3, help="Mean tolerance (relative)")
    ap.add_argument("--conv_std_tol", type=float, default=5e-2, help="Std tolerance (relative)")
    args = ap.parse_args()

    analyze_run(
        args.run_dir,
        out_subdir=args.out_subdir,
        smooth=args.smooth,
        hist_times=args.hist_times,
        conv_series=args.conv_series,
        conv_window=args.conv_window,
        conv_mean_tol=args.conv_mean_tol,
        conv_std_tol=args.conv_std_tol,
    )


if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) > 1:
        main()
    else:
        # Fallback ohne CLI-Argumente:
        # 1) ANALYZE_RUN_DIR (falls gesetzt)
        # 2) sonst: neuester Unterordner unter ./runs/mf_sac und ./runs/mf_td3
        override = os.environ.get("ANALYZE_RUN_DIR", "").strip()
        if override and os.path.isdir(override):
            run_dir = override
        else:
            roots = [
                os.path.join(".", "runs", "mf_sac"),
                os.path.join(".", "runs", "mf_td3"),
            ]
            candidates = []
            for root in roots:
                if os.path.isdir(root):
                    candidates += [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
            if not candidates:
                raise SystemExit(
                    "Kein Run-Ordner gefunden. Setze ANALYZE_RUN_DIR oder starte mit --run_dir."
                )
            run_dir = max(candidates, key=lambda p: os.path.getmtime(p))

        analyze_run(run_dir)
