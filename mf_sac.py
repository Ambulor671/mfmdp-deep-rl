# -*- coding: utf-8 -*-
"""
SAC components and training loop for MFCommonNoiseTriangleModel.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from mf_io import CSVLogger, ensure_dir, write_eval_row
from mf_triangle_modell import MFCommonNoiseTriangleModel
from mf_td3 import TD3_Replay  # reuse storage layout


class SAC_GaussianActor(nn.Module):
    def __init__(self, m: int, n: int, low: float, high: float, hidden=(256, 256),
                 log_std_min: float = -5.0, log_std_max: float = 1.0):
        super().__init__()
        h1, h2 = hidden
        self.fc1 = nn.Linear(m, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.mean = nn.Linear(h2, n)
        self.log_std = nn.Linear(h2, n)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.register_buffer("_low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("_high", torch.tensor(high, dtype=torch.float32))
        self.register_buffer("_mid", (self._low + self._high) / 2.0)
        self.register_buffer("_span", (self._high - self._low) / 2.0)

    def forward(self, mu: torch.Tensor):
        x = F.relu(self.fc1(mu))
        x = F.relu(self.fc2(x))

        m = self.mean(x)
        ls = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = torch.exp(ls)

        eps = torch.randn_like(m)
        pre = m + std * eps
        a_tanh = torch.tanh(pre)

        gauss = -0.5 * (((pre - m) / (std + 1e-8)) ** 2 + 2 * ls + np.log(2 * np.pi))
        tanh_j = -torch.log(1 - a_tanh.pow(2) + 1e-6)

        logp_per_dim = gauss + tanh_j
        logp_sum = logp_per_dim.sum(dim=-1, keepdim=True)

        a = self._mid + self._span * a_tanh
        mean_action = self._mid + self._span * torch.tanh(m)
        return a, logp_sum, mean_action, logp_per_dim

    @torch.no_grad()
    def act(self, mu: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        x = F.relu(self.fc1(mu))
        x = F.relu(self.fc2(x))
        m = self.mean(x)
        if deterministic:
            a_tanh = torch.tanh(m)
        else:
            ls = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
            a_tanh = torch.tanh(m + torch.exp(ls) * torch.randn_like(m))
        a = self._mid + self._span * a_tanh
        return torch.clamp(a, self._low, self._high)


class SAC_TwinCritic(nn.Module):
    def __init__(self, m: int, n: int, hidden=(256, 256)):
        super().__init__()
        self.q1_1 = nn.Linear(m + n, hidden[0])
        self.q1_2 = nn.Linear(hidden[0], hidden[1])
        self.q1_3 = nn.Linear(hidden[1], 1)

        self.q2_1 = nn.Linear(m + n, hidden[0])
        self.q2_2 = nn.Linear(hidden[0], hidden[1])
        self.q2_3 = nn.Linear(hidden[1], 1)

    def _mlp(self, x, l1, l2, l3):
        x = F.relu(l1(x))
        x = F.relu(l2(x))
        return l3(x)

    def forward(self, mu: torch.Tensor, a: torch.Tensor):
        x = torch.cat([mu, a], dim=-1)
        return (
            self._mlp(x, self.q1_1, self.q1_2, self.q1_3),
            self._mlp(x, self.q2_1, self.q2_2, self.q2_3),
        )

    def q_min(self, mu: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(mu, a)
        return torch.minimum(q1, q2)


class SAC_Replay(TD3_Replay):
    pass


@dataclass
class SAC_Config:
    episodes: int = 150
    T: int = 100
    batch_size: int = 64
    buffer_size: int = 100_000
    start_steps: int = 64
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    target_entropy: Optional[float] = None
    init_alpha: float = 0.2
    action_low: float = -1.0
    action_high: float = 1.0
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    grad_clip: float = 1.0
    eval_every: int = 10
    eval_episodes: int = 3
    device: str = "cpu"
    out_dir: str = "runs/mf_sac"
    run_name: str = "sac"
    seed: int = 49
    hidden: tuple[int, int] = (256, 256)
    weight_logp_by_mu: bool = True

    early_stop: bool = False
    es_window: int = 10
    es_rel_tol: float = 0.01
    es_std_tol: float = 0.5
    es_denom_eps: float = 1e-12


class SAC_Agent:
    def __init__(self, m: int, n: int, cfg: SAC_Config):
        self.m, self.n, self.cfg = m, n, cfg
        self.device = torch.device(cfg.device)

        self.actor = SAC_GaussianActor(
            m, n, cfg.action_low, cfg.action_high,
            hidden=cfg.hidden, log_std_min=cfg.log_std_min, log_std_max=cfg.log_std_max
        ).to(self.device)

        self.critic = SAC_TwinCritic(m, n, hidden=cfg.hidden).to(self.device)
        self.critic_targ = SAC_TwinCritic(m, n, hidden=cfg.hidden).to(self.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.opt_actor = Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.opt_alpha = Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.target_entropy = -float(n) if cfg.target_entropy is None else cfg.target_entropy

        self.total_updates = 0
        self.actor_loss_last = float("nan")
        self.critic_loss_last = float("nan")
        self.alpha_last = float(np.exp(self.log_alpha.detach().cpu().numpy()))
        self.logp_last = float("nan")
        self.q_pi_mean = float("nan")
        self.mean_std_last = float("nan")
        self.sum_log_std_last = float("nan")

        if cfg.weight_logp_by_mu:
            assert n == m, "weight_logp_by_mu=True expects n == m."

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, mu: np.ndarray, explore: bool = True) -> np.ndarray:
        mu_t = torch.from_numpy(mu).float().to(self.device).unsqueeze(0)
        a = self.actor.act(mu_t, deterministic=not explore)[0]
        return a.detach().cpu().numpy().astype(np.float32)

    def _polyak(self, net: nn.Module, targ: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p, pt in zip(net.parameters(), targ.parameters()):
                pt.data.mul_(1 - tau).add_(tau * p.data)

    def update(self, replay: SAC_Replay, gamma: float) -> None:
        cfg = self.cfg
        if replay.size < cfg.batch_size:
            return

        self.total_updates += 1

        mu, a, r, mu_next = replay.sample(cfg.batch_size)
        mu = mu.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        mu_next = mu_next.to(self.device)

        with torch.no_grad():
            a_next, logp_next_sum, _, logp_next_per_dim = self.actor(mu_next)
            y = self.critic_targ.q_min(mu_next, a_next)
            if cfg.weight_logp_by_mu:
                weighted_logp_next = (mu_next * logp_next_per_dim).sum(dim=-1, keepdim=True)
                y = y - self.alpha * weighted_logp_next
            else:
                y = y - self.alpha * logp_next_sum
            y = r + gamma * y

        q1, q2 = self.critic(mu, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.opt_critic.step()
        self.critic_loss_last = float(critic_loss.item())

        a_pi, logp_pi_sum, _, logp_pi_per_dim = self.actor(mu)
        q_pi = self.critic.q_min(mu, a_pi)

        if cfg.weight_logp_by_mu:
            weighted_logp = (mu * logp_pi_per_dim).sum(dim=-1, keepdim=True)
            actor_loss = (self.alpha * weighted_logp - q_pi).mean()
            cur_logp_for_log = weighted_logp
        else:
            actor_loss = (self.alpha * logp_pi_sum - q_pi).mean()
            cur_logp_for_log = logp_pi_sum

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.opt_actor.step()
        self.actor_loss_last = float(actor_loss.item())

        alpha_loss = -(self.log_alpha * (logp_pi_sum.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        self._polyak(self.critic, self.critic_targ, cfg.tau)

        self.alpha_last = float(self.alpha.detach().cpu().item())
        self.q_pi_mean = float(q_pi.mean().item())
        self.logp_last = float(cur_logp_for_log.mean().item())

        with torch.no_grad():
            x = F.relu(self.actor.fc1(mu))
            x = F.relu(self.actor.fc2(x))
            log_std_b = torch.clamp(self.actor.log_std(x), self.actor.log_std_min, self.actor.log_std_max)
            std_b = torch.exp(log_std_b)
            self.mean_std_last = float(std_b.mean().item())
            self.sum_log_std_last = float(log_std_b.sum(dim=-1).mean().item())


def train_sac(
    model: MFCommonNoiseTriangleModel,
    cfg: SAC_Config,
    eval_fn,  # injected to avoid a hard dependency on compare/eval module
    eval_seeds: Optional[List[int]] = None,
    eval_model_params: Optional[Dict[str, Any]] = None,
    eval_T: Optional[int] = None,
    eval_gamma: Optional[float] = None,
    on_eval: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Tuple[SAC_Agent, str]:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    assert abs(model.gamma - cfg.gamma) < 1e-9

    ensure_dir(cfg.out_dir)
    run_dir = os.path.join(cfg.out_dir, f"{cfg.run_name}_{int(time.time())}")
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    agent = SAC_Agent(model.m, model.n, cfg)
    replay = SAC_Replay(model.m, model.n, cfg.buffer_size)

    curve_csv = os.path.join(run_dir, "eval_curve.csv")

    step_logger = CSVLogger(
        os.path.join(run_dir, "train_steps.csv"),
        ["episode", "t", "reward", "stage_cost", "mu_delta_sq",
         "actor_loss", "critic_loss", "alpha", "logp_pi", "q_pi_mean",
         "mean_std", "sum_log_std"],
    )
    ep_logger = CSVLogger(
        os.path.join(run_dir, "train_episodes.csv"),
        ["episode", "sum_stage_cost", "avg_stage_cost", "discounted_cost", "return_reward"],
    )
    eval_logger = CSVLogger(
        os.path.join(run_dir, "eval.csv"),
        ["episode", "mean_J_return", "mean_discounted_cost", "episodes"],
    )

    total_steps = 0
    ep_disc_costs: List[float] = []
    first_update_printed = False

    for ep in range(cfg.episodes):
        if ep % 10 == 0:
            print(f"Episode {ep}")

        mu = model.reset()
        ep_stage_cost_sum = 0.0
        ep_rewards_sum = 0.0
        ep_disc_cost = 0.0
        gamma_pow = 1.0

        for t in range(cfg.T):
            a = agent.act(mu, explore=True)
            mu_next, r, info = model.step(mu, a, train=True)

            stage_cost = float(info.get("stage_cost", -float(r)))
            mu_delta_sq = float(np.sum((mu_next - mu) ** 2))

            replay.push(mu, a, float(r), mu_next)
            total_steps += 1

            did_update = False
            if (total_steps >= cfg.start_steps) and (replay.size >= cfg.batch_size):
                if not first_update_printed:
                    print(f"[info] start updates at env_step={total_steps}, replay_size={replay.size}")
                    first_update_printed = True
                for _ in range(cfg.updates_per_step):
                    agent.update(replay, gamma=cfg.gamma)
                    did_update = True

            step_logger.log({
                "episode": ep,
                "t": t,
                "reward": float(r),
                "stage_cost": stage_cost,
                "mu_delta_sq": mu_delta_sq,
                "actor_loss": agent.actor_loss_last if did_update else float("nan"),
                "critic_loss": agent.critic_loss_last if did_update else float("nan"),
                "alpha": agent.alpha_last if did_update else float("nan"),
                "logp_pi": agent.logp_last if did_update else float("nan"),
                "q_pi_mean": agent.q_pi_mean if did_update else float("nan"),
                "mean_std": agent.mean_std_last if did_update else float("nan"),
                "sum_log_std": agent.sum_log_std_last if did_update else float("nan"),
            })

            ep_stage_cost_sum += stage_cost
            ep_rewards_sum += float(r)
            ep_disc_cost += gamma_pow * stage_cost
            gamma_pow *= cfg.gamma
            mu = mu_next

        ep_logger.log({
            "episode": ep,
            "sum_stage_cost": ep_stage_cost_sum,
            "avg_stage_cost": ep_stage_cost_sum / cfg.T,
            "discounted_cost": ep_disc_cost,
            "return_reward": ep_rewards_sum,
        })
        ep_disc_costs.append(float(ep_disc_cost))

        if eval_seeds and ((ep + 1) % cfg.eval_every == 0):
            def _pol(mu_arr, deterministic=True):
                return agent.act(mu_arr, explore=not deterministic)

            end_vals, Jevals, _, _ = eval_fn(
                eval_model_params if eval_model_params is not None else dict(gamma=cfg.gamma, m=model.m),
                eval_T if eval_T is not None else cfg.T,
                _pol, deterministic=True, seeds=eval_seeds,
                gamma=(eval_gamma if eval_gamma is not None else cfg.gamma),
                return_traj=False
            )

            row = {
                "episode": ep + 1,
                "end_mean": float(end_vals.mean()),
                "end_std": float(end_vals.std(ddof=1)) if len(end_vals) > 1 else 0.0,
                "Jeval_mean": float(Jevals.mean()),
                "Jeval_std": float(Jevals.std(ddof=1)) if len(Jevals) > 1 else 0.0,
                "seed_n": int(len(eval_seeds)),
            }
            write_eval_row(curve_csv, row)
            if on_eval is not None:
                on_eval(ep + 1, row)

            eval_logger.log({
                "episode": ep,
                "mean_J_return": float(-Jevals.mean()),
                "mean_discounted_cost": float(Jevals.mean()),
                "episodes": int(len(eval_seeds)),
            })

        if cfg.early_stop:
            W = int(cfg.es_window)
            if W > 0 and len(ep_disc_costs) >= 2 * W:
                prev = np.asarray(ep_disc_costs[-2 * W:-W], dtype=float)
                curr = np.asarray(ep_disc_costs[-W:], dtype=float)
                mean_prev = float(prev.mean())
                mean_curr = float(curr.mean())
                std_curr = float(curr.std(ddof=1)) if W > 1 else 0.0
                rel_change = abs(mean_curr - mean_prev) / max(abs(mean_prev), cfg.es_denom_eps)
                if (rel_change < cfg.es_rel_tol) and (std_curr < cfg.es_std_tol):
                    break

    step_logger.close()
    ep_logger.close()
    eval_logger.close()
    return agent, run_dir
