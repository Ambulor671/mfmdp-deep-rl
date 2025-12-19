# -*- coding: utf-8 -*-
"""
mf_triangle_model.py

Discrete triangle mean-field environment with:
- transport: "spread" | "wta" | "softwta"
- dyn_mode:  "base" | "meanpush" | "densitygrad"
- noise switches:
    * use_common_noise: shared shift e0 = J + Z
    * use_idio_noise:  additional diffusion / smoothing in the spread transport
      with idio_model = "gauss" (action-independent) or "a_gauss" (action-dependent)

Notes:
- In this implementation, diffusion acts only in the "spread" transport.
  For "wta" and "softwta", the update remains (mostly) point-assignment based.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
import math


class MFCommonNoiseTriangleModel:
    MU_TARGET_DEFAULT = np.array(
        [0.0, 0.0, 0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05, 0.0, 0.0],
        dtype=np.float64
    )

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        seed: int = 3,
        mu_floor: float = 1e-9,

        # --- common noise e0 = J + Z ---
        use_common_noise: bool = True,
        p_jump: float = 0.05,
        sigma: float = 0.2,

        # --- idiosyncratic diffusion (spread transport) ---
        use_idio_noise: bool = False,
        idio_model: str = "gauss",      # "gauss" | "a_gauss"
        id_p_jump: float = 0.0,         # optional +/-1 mixture
        id_sigma: float = 0.0,          # additional Gaussian width
        id_radius: Optional[int] = None,

        # --- action-dependent width sigma(a) for idio_model="a_gauss" ---
        ad_sigma_base: float = 0.05,
        ad_sigma_gain: float = 0.45,
        ad_sigma_min: float = 0.0,
        ad_sigma_max: Optional[float] = 1.5,

        # --- costs ---
        action_penalty: str = "abs",    # "abs" | "l2" | "sqrt"
        penalty_eps: float = 1e-6,

        # --- transport ---
        transport: str = "spread",      # "spread" | "wta" | "softwta"
        soft_tau: float = 0.1,

        # --- target / grid ---
        mu_target: Optional[np.ndarray] = None,
        m: int = 11,

        # --- dynamics mode ---
        dyn_mode: str = "base",         # "base" | "meanpush" | "densitygrad"
        beta: float = 0.05,

        # --- densitygrad options ---
        grad_sigma: float = 0.8,
        grad_radius: Optional[int] = None,
        grad_clip: float = 0.6,
        grad_normalize: bool = False,
    ):
        self.m = int(m)
        self.n = self.m
        self.gamma = float(gamma)
        self.mu_floor = float(mu_floor)

        # Common noise
        self.use_common_noise = bool(use_common_noise)
        self.p_jump = float(p_jump)
        self.sigma = float(sigma)

        # Idio noise
        self.use_idio_noise = bool(use_idio_noise)
        self.idio_model = str(idio_model).lower()
        if self.idio_model not in {"gauss", "a_gauss"}:
            raise ValueError("idio_model must be 'gauss' or 'a_gauss'")
        self.id_p_jump = float(id_p_jump)
        self.id_sigma = float(id_sigma)
        if id_radius is None:
            self.id_radius = int(math.ceil(3.0 * max(0.0, self.id_sigma)))
        else:
            self.id_radius = int(max(0, id_radius))

        # a-dependent sigma(a)
        self.ad_sigma_base = float(ad_sigma_base)
        self.ad_sigma_gain = float(ad_sigma_gain)
        self.ad_sigma_min = float(ad_sigma_min)
        self.ad_sigma_max = None if ad_sigma_max is None else float(ad_sigma_max)

        # Penalty
        self.action_penalty = str(action_penalty).lower()
        self.penalty_eps = float(penalty_eps)

        # Transport
        self.transport = str(transport).lower()
        if self.transport not in {"spread", "wta", "softwta"}:
            raise ValueError("transport must be 'spread'|'wta'|'softwta'")
        self.soft_tau = float(soft_tau)

        # Dynamics
        self.dyn_mode = str(dyn_mode).lower()
        if self.dyn_mode not in {"base", "meanpush", "densitygrad"}:
            raise ValueError("dyn_mode must be 'base'|'meanpush'|'densitygrad'")
        self.beta = float(beta)
        self.grad_sigma = float(grad_sigma)
        if grad_radius is None:
            self.grad_radius = int(math.ceil(3.0 * max(0.0, self.grad_sigma)))
        else:
            self.grad_radius = int(max(0, grad_radius))
        self.grad_clip = float(grad_clip)
        self.grad_normalize = bool(grad_normalize)

        self._rng = np.random.RandomState(seed)

        # Target distribution
        if mu_target is None:
            if len(self.MU_TARGET_DEFAULT) == self.m:
                self.mu_target = self.MU_TARGET_DEFAULT.copy()
            else:
                self.mu_target = np.full(self.m, 1.0 / self.m, dtype=np.float64)
        else:
            mu_target = np.asarray(mu_target, dtype=np.float64)
            if mu_target.shape != (self.m,):
                raise ValueError(f"mu_target must have shape ({self.m},)")
            s = float(mu_target.sum())
            self.mu_target = mu_target / (s if s > 0 else 1.0)

        # State-dependent action bounds
        self._a_low = np.full(self.m, -1.0, dtype=np.float32)
        self._a_high = np.full(self.m, 1.0, dtype=np.float32)
        self._a_low[0] = 0.0
        self._a_high[-1] = 0.0

    # ---------------- utilities ----------------
    @staticmethod
    def _ensure_prob(v: np.ndarray, eps: float) -> np.ndarray:
        v = np.clip(v, eps, None)
        v = v / np.sum(v, dtype=np.float64)
        return v.astype(np.float32)

    @staticmethod
    def _linspread_add_clipped(dst: np.ndarray, pos: float, mass: float) -> None:
        m = dst.shape[0]
        if pos <= 0.0:
            dst[0] += mass
            return
        if pos >= m - 1:
            dst[m - 1] += mass
            return
        i0 = int(np.floor(pos))
        frac = pos - i0
        i1 = i0 + 1
        dst[i0] += (1.0 - frac) * mass
        dst[i1] += frac * mass

    def _gauss_splat_add_clipped(self, dst: np.ndarray, center: float, mass: float, sigma: float, radius: int) -> None:
        if sigma <= 0.0 or radius <= 0:
            self._linspread_add_clipped(dst, center, mass)
            return
        m = dst.shape[0]
        j_min = max(0, int(math.floor(center - radius)))
        j_max = min(m - 1, int(math.ceil(center + radius)))
        if j_min > j_max:
            self._linspread_add_clipped(dst, center, mass)
            return
        js = np.arange(j_min, j_max + 1, dtype=np.float64)
        w = np.exp(-0.5 * ((js - center) ** 2) / (sigma ** 2))
        s = float(np.sum(w))
        if s <= 0.0 or not np.isfinite(s):
            self._linspread_add_clipped(dst, center, mass)
            return
        w = w / s
        dst[j_min:j_max + 1] += mass * w

    def _soft_assign(self, dst: np.ndarray, center: float, mass: float, tau: float) -> None:
        m = dst.shape[0]
        j0 = int(np.round(np.clip(center, 0, m - 1)))
        js = [j0]
        if j0 > 0:
            js.append(j0 - 1)
        if j0 < m - 1:
            js.append(j0 + 1)
        js = sorted(set(js))
        scores = np.array([-abs(j - center) / max(tau, 1e-6) for j in js], dtype=float)
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        for j, wj in zip(js, w):
            dst[j] += mass * float(wj)

    def _sample_common_noise(self) -> float:
        jump = 0.0
        if self._rng.rand() < self.p_jump:
            jump = 1.0 if (self._rng.rand() < 0.5) else -1.0
        z = self._rng.randn() * self.sigma
        return float(jump + z)

    def _sigma_of_a(self, a_val: float) -> float:
        s = self.ad_sigma_base + self.ad_sigma_gain * abs(float(a_val))
        if self.ad_sigma_max is not None:
            s = min(s, self.ad_sigma_max)
        s = max(self.ad_sigma_min, s)
        return float(s)

    def _smooth_density(self, mu: np.ndarray) -> np.ndarray:
        if self.grad_sigma <= 0.0 or self.grad_radius <= 0:
            return mu
        m = mu.shape[0]
        out = np.zeros_like(mu, dtype=np.float64)
        js = np.arange(m, dtype=np.float64)
        for i in range(m):
            j_min = max(0, int(i - self.grad_radius))
            j_max = min(m - 1, int(i + self.grad_radius))
            jseg = js[j_min:j_max + 1]
            w = np.exp(-0.5 * ((jseg - i) ** 2) / (self.grad_sigma ** 2))
            sw = float(np.sum(w))
            if sw > 0.0 and np.isfinite(sw):
                out[i] = float(np.dot(w, mu[j_min:j_max + 1]) / sw)
            else:
                out[i] = mu[i]
        s = float(out.sum())
        if s > 0:
            out = out / s
        return out

    def _density_gradient(self, mu: np.ndarray) -> np.ndarray:
        rho = self._smooth_density(mu.astype(np.float64))
        m = rho.shape[0]
        grad = np.zeros(m, dtype=np.float64)
        if m == 1:
            return grad
        grad[1:-1] = 0.5 * (rho[2:] - rho[:-2])
        grad[0] = rho[1] - rho[0]
        grad[-1] = rho[-1] - rho[-2]

        if self.grad_normalize:
            gmax = float(np.max(np.abs(grad)))
            if gmax > 1e-12:
                grad = grad / gmax

        if self.grad_clip > 0.0:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)
        return grad

    # ---------------- API ----------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        mu0 = self._rng.rand(self.m).astype(np.float32)
        return self._ensure_prob(mu0, self.mu_floor)

    def step(self, mu: np.ndarray, a: np.ndarray, *, train: bool) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        mu = np.asarray(mu, dtype=np.float32)
        a_in = np.asarray(a, dtype=np.float32).reshape(-1)
        if a_in.size < self.m:
            a_in = np.pad(a_in, (0, self.m - a_in.size))
        a_in = a_in[: self.m]
        a_clipped = np.clip(a_in, self._a_low, self._a_high)

        # --- stage cost / reward ---
        mu_safe = np.clip(mu, self.mu_floor, None).astype(np.float64)
        aa = np.abs(a_clipped).astype(np.float64)
        if self.action_penalty == "abs":
            phi = aa
        elif self.action_penalty == "l2":
            phi = aa ** 2
        elif self.action_penalty == "sqrt":
            phi = np.sqrt(aa + self.penalty_eps)
        else:
            raise ValueError(f"unknown action_penalty={self.action_penalty}")
        E_phi = float(np.sum(mu_safe * phi))
        diff = mu_safe - self.mu_target
        l2_sq = float(np.sum(diff * diff))
        stage_cost = E_phi + l2_sq
        reward = -stage_cost

        # --- common noise ---
        e0 = self._sample_common_noise() if self.use_common_noise else 0.0

        # --- dynamics extra drift ---
        if self.dyn_mode == "base":
            extra_by_i = np.zeros(self.m, dtype=np.float64)
            dyn_info = {"dyn_mode": "base"}
        elif self.dyn_mode == "meanpush":
            idx = np.arange(self.m, dtype=np.float64)
            mean_s = float(np.dot(idx, mu_safe))
            extra = self.beta * mean_s
            extra_by_i = np.full(self.m, extra, dtype=np.float64)
            dyn_info = {"dyn_mode": "meanpush", "mean_s": mean_s, "beta": self.beta, "extra": extra}
        else:
            grad = self._density_gradient(mu_safe)
            extra_by_i = -self.beta * grad
            dyn_info = {
                "dyn_mode": "densitygrad",
                "beta": self.beta,
                "grad_max_abs": float(np.max(np.abs(grad))) if grad.size else 0.0,
                "grad_sigma": self.grad_sigma,
                "grad_clip": self.grad_clip,
                "grad_normalize": self.grad_normalize,
            }

        # --- idiosyncratic diffusion configuration ---
        use_idio = bool(self.use_idio_noise)
        if use_idio:
            use_idio = (self.id_sigma > 0.0) or (self.id_p_jump > 0.0) or (self.idio_model == "a_gauss")

        pj = float(self.id_p_jump) if use_idio else 0.0
        w_center = max(0.0, 1.0 - pj)
        w_side = 0.5 * max(0.0, min(1.0, pj))

        mu_next = np.zeros(self.m, dtype=np.float64)

        for i in range(self.m):
            mass = float(mu_safe[i])
            if mass <= 0.0:
                continue

            base_center = float(i) + float(a_clipped[i]) + float(extra_by_i[i]) + float(e0)

            if self.transport == "spread":
                if not use_idio:
                    self._linspread_add_clipped(mu_next, base_center, mass)
                    continue

                ai = float(a_clipped[i])
                if self.idio_model == "gauss":
                    sigma_tot = float(self.id_sigma)
                else:
                    sigma_a = self._sigma_of_a(ai)
                    sigma_tot = math.sqrt(max(0.0, sigma_a**2 + self.id_sigma**2))

                rad = int(max(1, math.ceil(3.0 * sigma_tot))) if sigma_tot > 0 else self.id_radius

                jump_mix = [(0.0, w_center)]
                if w_side > 0.0:
                    jump_mix += [(-1.0, w_side), (+1.0, w_side)]

                for jmp, wj in jump_mix:
                    if wj <= 0:
                        continue
                    self._gauss_splat_add_clipped(
                        mu_next, base_center + float(jmp), mass * float(wj), sigma_tot, rad
                    )

            elif self.transport == "wta":
                j = int(np.round(base_center))
                j = 0 if j < 0 else (self.m - 1 if j > self.m - 1 else j)
                mu_next[j] += mass

            else:
                self._soft_assign(mu_next, base_center, mass, tau=self.soft_tau)

        mu_next = self._ensure_prob(mu_next, self.mu_floor)

        info: Dict[str, Any] = {
            "difference_to_target": float(np.sum((mu_next - self.mu_target) ** 2)),
            "stage_cost": stage_cost,
            "reward": reward,
            "E_abs_a": E_phi,
            "L2_mu_to_target_sq": l2_sq,
            "train": bool(train),

            "transport": self.transport,
            "soft_tau": self.soft_tau,

            "dyn_mode": dyn_info.get("dyn_mode", self.dyn_mode),
            "use_common_noise": bool(self.use_common_noise),
            "e0": float(e0),
            "p_jump": self.p_jump,
            "sigma": self.sigma,

            "use_idio_noise": bool(self.use_idio_noise),
            "idio_model": self.idio_model,
            "id_p_jump": self.id_p_jump,
            "id_sigma": self.id_sigma,
            "id_radius": self.id_radius,
        }
        info.update(dyn_info)

        if self.idio_model == "a_gauss":
            info["mean_sigma_of_a"] = float(np.mean([self._sigma_of_a(x) for x in a_clipped]))
            info["ad_sigma_base"] = self.ad_sigma_base
            info["ad_sigma_gain"] = self.ad_sigma_gain
            info["ad_sigma_min"] = self.ad_sigma_min
            info["ad_sigma_max"] = self.ad_sigma_max if self.ad_sigma_max is not None else None

        return mu_next, reward, info


