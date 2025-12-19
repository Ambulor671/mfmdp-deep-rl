# -*- coding: utf-8 -*-
"""
TD3 components and training loop for MFCommonNoiseTriangleModel.
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


class TD3_MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(256, 256), act_last: bool = True):
        super().__init__()
        h1, h2 = hidden
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)
        self.act_last = act_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.act_last:
            x = torch.tanh(x)
        return x


class TD3_Actor(nn.Module):
    def __init__(self, m: int, n: int, hidden=(256, 256)):
        super().__init__()
        self.net = TD3_MLP(m, n, hidden=hidden, act_last=True)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu)


class TD3_Critic(nn.Module):
    def __init__(self, m: int, n: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = TD3_MLP(m + n, 1, hidden=hidden, act_last=False)
        self.q2 = TD3_MLP(m + n, 1, hidden=hidden, act_last=False)

    def forward(self, mu: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([mu, a], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, mu: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mu, a], dim=-1)
        return self.q1(x)


class TD3_Replay:
    def __init__(self, m: int, n: int, capacity: int):
        self.m, self.n = m, n
        self.capacity = int(capacity)
        self.mu = np.zeros((self.capacity, m), dtype=np.float32)
        self.a = np.zeros((self.capacity, n), dtype=np.float32)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.mu_next = np.zeros((self.capacity, m), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def push(self, mu: np.ndarray, a: np.ndarray, r: float, mu_next: np.ndarray) -> None:
        i = self.ptr
        self.mu[i] = mu
        self.a[i] = a
        self.r[i] = r
        self.mu_next[i] = mu_next
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        mu = torch.from_numpy(self.mu[idx])
        a = torch.from_numpy(self.a[idx])
        r = torch.from_numpy(self.r[idx]).unsqueeze(-1)
        mu_next = torch.from_numpy(self.mu_next[idx])
        return mu, a, r, mu_next


@dataclass
class TD3_Config:
    episodes: int = 150
    T: int = 100
    batch_size: int = 64
    buffer_size: int = 100_000
    start_steps: int = 64
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise_sigma: float = 0.20
    noise_clip: float = 0.20
    policy_delay: int = 2
    explore_sigma: float = 0.01
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    grad_clip: float = 1.0
    action_low: float = -1.0
    action_high: float = 1.0
    eval_every: int = 10
    eval_episodes: int = 3
    device: str = "cpu"
    out_dir: str = "runs/mf_td3"
    run_name: str = "td3"
    seed: int = 44
    hidden: tuple[int, int] = (256, 256)

    early_stop: bool = False
    es_window: int = 10
    es_rel_tol: float = 0.01
    es_std_tol: float = 0.5
    es_denom_eps: float = 1e-12


class TD3_Agent:
    def __init__(self, m: int, n: int, cfg: TD3_Config):
        self.m, self.n, self.cfg = m, n, cfg
        self.device = torch.device(cfg.device)

        self.actor = TD3_Actor(m, n, hidden=cfg.hidden).to(self.device)
        self.actor_targ = TD3_Actor(m, n, hidden=cfg.hidden).to(self.device)
        self.critic = TD3_Critic(m, n, hidden=cfg.hidden).to(self.device)
        self.critic_targ = TD3_Critic(m, n, hidden=cfg.hidden).to(self.device)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.opt_actor = Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self._low_t = torch.tensor(cfg.action_low, dtype=torch.float32, device=self.device)
        self._high_t = torch.tensor(cfg.action_high, dtype=torch.float32, device=self.device)

        self.total_updates = 0
        self.actor_loss_last = float("nan")
        self.critic_loss_last = float("nan")

    @torch.no_grad()
    def act(self, mu: np.ndarray, explore: bool = True) -> np.ndarray:
        mu_t = torch.from_numpy(mu).float().to(self.device).unsqueeze(0)
        a = self.actor(mu_t)[0]
        if explore:
            a = torch.clamp(a + torch.randn_like(a) * self.cfg.explore_sigma, -1.0, 1.0)
        else:
            a = torch.clamp(a, -1.0, 1.0)
        a = torch.clamp(a, self._low_t, self._high_t)
        return a.cpu().numpy().astype(np.float32)

    def _polyak(self, net: nn.Module, targ: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p, pt in zip(net.parameters(), targ.parameters()):
                pt.data.mul_(1 - tau).add_(tau * p.data)

    def update(self, replay: TD3_Replay, gamma: float) -> None:
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
            a_targ = self.actor_targ(mu_next)
            noise = torch.randn_like(a_targ) * cfg.policy_noise_sigma
            noise = torch.clamp(noise, -cfg.noise_clip, cfg.noise_clip)
            a_targ = torch.clamp(a_targ + noise, -1.0, 1.0)
            a_targ = torch.clamp(a_targ, self._low_t, self._high_t)
            q1_t, q2_t = self.critic_targ(mu_next, a_targ)
            y = r + gamma * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(mu, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.opt_critic.step()
        self.critic_loss_last = float(critic_loss.item())

        if self.total_updates % cfg.policy_delay == 0:
            a_pred = torch.clamp(self.actor(mu), -1.0, 1.0)
            a_pred = torch.clamp(a_pred, self._low_t, self._high_t)
            actor_loss = -self.critic.q1_only(mu, a_pred).mean()

            self.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
            self.opt_actor.step()
            self.actor_loss_last = float(actor_loss.item())

            self._polyak(self.critic, self.critic_targ, cfg.tau)
            self._polyak(self.actor, self.actor_targ, cfg.tau)


def train_td3(
    model: MFCommonNoiseTriangleModel,
    cfg: TD3_Config,
    eval_fn,
    eval_seeds: Optional[List[int]] = None,
    eval_model_params: Optional[Dict[str, Any]] = None,
    eval_T: Optional[int] = None,
    eval_gamma: Optional[float] = None,
    on_eval: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Tuple[TD3_Agent, str]:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    assert abs(model.gamma - cfg.gamma) < 1e-9

    ensure_dir(cfg.out_dir)
    run_dir = os.path.join(cfg.out_dir, f"{cfg.run_name}_{int(time.time())}")
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    agent = TD3_Agent(model.m, model.n, cfg)
    replay = TD3_Replay(model.m, model.n, cfg.buffer_size)

    step_logger = CSVLogger(
        os.path.join(run_dir, "train_steps.csv"),
        ["episode", "t", "reward", "stage_cost", "mu_delta_sq", "actor_loss", "critic_loss"],
    )
    ep_logger = CSVLogger(
        os.path.join(run_dir, "train_episodes.csv"),
        ["episode", "sum_stage_cost", "avg_stage_cost", "discounted_cost", "return_reward"],
    )
    eval_logger = CSVLogger(
        os.path.join(run_dir, "eval.csv"),
        ["episode", "mean_J_return", "mean_discounted_cost", "episodes"],
    )

    curve_csv = os.path.join(run_dir, "eval_curve.csv")
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
