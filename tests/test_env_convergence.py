"""Delta-hedge loss std halves when steps quadruple (vanilla call, GBM, zero cost)."""

import math

import torch

from niftyhedge.hedge.contracts import Contract, barrier_payoff
from niftyhedge.hedge.instruments import CostModel
from niftyhedge.utils.seed import set_seed


def _gbm_paths(n: int, steps: int, S0: float, sigma: float, r: float, seed: int):
    set_seed(seed)
    dt = 1.0 / steps  # unit time
    S = torch.zeros(n, steps + 1)
    S[:, 0] = S0
    z = torch.randn(n, steps)
    for t in range(steps):
        S[:, t + 1] = S[:, t] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z[:, t])
    return S, dt


def _bs_delta(S, K, T, r, sigma):
    T = torch.clamp(T, min=1e-8)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    return 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))


def _hedge_loss_std(steps: int, n: int = 4000) -> float:
    S0, K, sigma, r = 100.0, 100.0, 0.2, 0.0
    S, dt = _gbm_paths(n, steps, S0, sigma, r, seed=0)
    H = steps
    h = torch.zeros(n, H)
    for t in range(H):
        rem = (H - t) * dt
        T = torch.full((n,), rem)
        h[:, t] = _bs_delta(S[:, t], K, T, r, sigma)
    dS = S[:, 1:] - S[:, :-1]
    pnl = (h * dS).sum(dim=1)
    payoff = torch.clamp(S[:, -1] - K, min=0.0)
    # BS price at inception
    from niftyhedge.data.iv import bs_price
    import numpy as np

    p0 = float(bs_price(S0, K, 1.0, r, sigma, "call"))
    L = payoff - p0 - pnl
    return float(L.std())


def test_loss_std_halves_when_steps_quadruple():
    s1 = _hedge_loss_std(16)
    s4 = _hedge_loss_std(64)
    # theoretical: std ~ 1/sqrt(n_steps); 4x steps => ~0.5x std
    ratio = s4 / s1
    assert ratio < 0.7, f"expected ~0.5 ratio, got {ratio} (s1={s1}, s4={s4})"
