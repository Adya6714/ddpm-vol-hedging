"""W1: Heston world with COS pricer and simple calibration."""

from __future__ import annotations

import math

import numpy as np
import torch

from niftyhedge.data.splits import FoldData
from niftyhedge.data.surface import MONEYNESS, TENORS_DAYS
from niftyhedge.gen.base import Paths, StartStates
from niftyhedge.utils.seed import set_seed


def heston_char_logS(u, S, T, r, kappa, theta, xi, rho, v0):
    """Characteristic function of x = log(S_T)."""
    iu = 1j * u
    a = kappa * theta
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    exp_dt = np.exp(-d * T)
    C = r * iu * T + (a / xi**2) * ((b - d) * T - 2 * np.log((1 - g * exp_dt) / (1 - g)))
    D = ((b - d) / xi**2) * ((1 - exp_dt) / (1 - g * exp_dt))
    return np.exp(C + D * v0 + iu * np.log(S))


def cos_call_price(S, K, T, r, kappa, theta, xi, rho, v0, N=128, L=10.0) -> float:
    """Fang–Oosterlee COS European call on x = log(S_T)."""
    if T <= 1e-12:
        return float(max(S - K, 0.0))
    # cumulants of x = log(S_T)
    c1 = np.log(S) + (r - 0.5 * theta) * T + (theta - v0) * (1 - np.exp(-kappa * T)) / (2 * kappa)
    c2 = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
    c2 = max(float(np.real(c2)), 1e-6)
    a = float(c1 - L * np.sqrt(c2))
    b = float(c1 + L * np.sqrt(c2))
    k = np.arange(N)
    u = k * np.pi / (b - a)
    phi = heston_char_logS(u, S, T, r, kappa, theta, xi, rho, v0)

    def chi(k_arr, c, d):
        kp = k_arr * np.pi / (b - a)
        num = (
            np.cos(kp * (d - a)) * np.exp(d)
            - np.cos(kp * (c - a)) * np.exp(c)
            + kp * (np.sin(kp * (d - a)) * np.exp(d) - np.sin(kp * (c - a)) * np.exp(c))
        )
        return num / (1 + kp * kp)

    def psi(k_arr, c, d):
        kp = k_arr * np.pi / (b - a)
        out = np.empty_like(kp, dtype=float)
        out[k_arr == 0] = d - c
        nz = k_arr != 0
        out[nz] = (np.sin(kp[nz] * (d - a)) - np.sin(kp[nz] * (c - a))) / kp[nz]
        return out

    kk = np.log(K)
    c, d = max(kk, a), b
    if d <= c:
        return 0.0
    Vk = 2.0 / (b - a) * (chi(k, c, d) - K * psi(k, c, d))
    Fk = np.real(phi * np.exp(-1j * k * np.pi * a / (b - a)))
    Fk[0] *= 0.5
    price = float(np.exp(-r * T) * np.sum(Fk * Vk))
    return max(price, 0.0)


def heston_iv_surface(S, r, kappa, theta, xi, rho, v0) -> np.ndarray:
    from niftyhedge.data.iv import implied_vol

    surf = np.zeros((9, 6))
    for j, tau in enumerate(TENORS_DAYS):
        T = tau / 365.0
        for i, m in enumerate(MONEYNESS):
            K = m * S
            px = cos_call_price(S, K, T, r, kappa, theta, xi, rho, v0)
            iv = implied_vol(np.array([px]), np.array([S]), np.array([K]), np.array([T]), r, "call")
            val = float(np.asarray(iv).ravel()[0])
            surf[i, j] = val if np.isfinite(val) else float(np.sqrt(max(theta, 1e-8)))
    return np.clip(surf, 0.05, 1.5)


class HestonWorld:
    name = "heston"

    def __init__(self, r: float = 0.065, dt: float = 1.0 / 252.0):
        self.r = r
        self.dt = dt
        self.params = dict(kappa=2.0, theta=0.04, xi=0.3, rho=-0.6)

    def fit(self, fold: FoldData, seed: int) -> None:
        atm = fold.X_train[:, 4, 2]
        self.params["theta"] = float(np.clip(np.mean(atm**2), 0.01, 0.25))
        self._fitted = True

    def sample_paths(self, start_states: StartStates, n_paths: int, horizon: int, seed: int) -> Paths:
        set_seed(seed)
        device = start_states.spot.device
        n_start = start_states.spot.shape[0]
        pick = torch.randint(0, n_start, (n_paths,), device=device)
        S0 = start_states.spot[pick]
        surf0 = start_states.surf[pick]
        idx = start_states.idx[pick]
        v0 = (surf0[:, 4, 2] ** 2).clamp(1e-4, 1.0)

        kappa, theta, xi, rho = (
            self.params["kappa"],
            self.params["theta"],
            self.params["xi"],
            self.params["rho"],
        )
        S = torch.zeros(n_paths, horizon + 1, device=device)
        v = torch.zeros(n_paths, horizon + 1, device=device)
        S[:, 0] = S0
        v[:, 0] = v0
        z1 = torch.randn(n_paths, horizon, device=device)
        z2 = torch.randn(n_paths, horizon, device=device)
        z_v = rho * z1 + math.sqrt(max(1 - rho**2, 0.0)) * z2
        for t in range(horizon):
            v_pos = v[:, t].clamp(min=0.0)
            v[:, t + 1] = torch.clamp(
                v_pos
                + kappa * (theta - v_pos) * self.dt
                + xi * torch.sqrt(v_pos * self.dt) * z_v[:, t],
                min=1e-8,
            )
            S[:, t + 1] = S[:, t] * torch.exp(
                (self.r - 0.5 * v_pos) * self.dt + torch.sqrt(v_pos * self.dt) * z1[:, t]
            )

        surf = torch.zeros(n_paths, horizon + 1, 9, 6, device=device)
        atm0 = surf0[:, 4, 2].clamp(min=1e-4)
        for t in range(horizon + 1):
            factor = (v[:, t].clamp(min=1e-8).sqrt() / atm0).clamp(0.25, 4.0)
            surf[:, t] = (surf0 * factor[:, None, None]).clamp(0.05, 1.5)

        paths = Paths(S=S, surf=surf, start_idx=idx)
        paths.validate()
        return paths
