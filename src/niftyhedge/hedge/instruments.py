"""Vanilla option marks from the path surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import norm

from niftyhedge.data.surface import MONEYNESS, TENORS_DAYS


@dataclass
class CostModel:
    underlying_bps: float = 2.0
    option_frac: float = 0.01  # fraction of premium


@dataclass
class InstrumentSet:
    """Fixed strikes/expiries chosen at inception; n_opt vanillas."""

    kinds: list[str]  # "call" / "put"
    strike_mults: list[float]  # relative to S0
    tenor_days: list[int]


def lookup_iv(
    surf: torch.Tensor,
    moneyness: float | torch.Tensor,
    tau_days: float | torch.Tensor,
) -> torch.Tensor:
    """Bilinear interp of IV on 9x6 grid. surf: [..., 9, 6]."""
    m_grid = torch.as_tensor(MONEYNESS, dtype=surf.dtype, device=surf.device)
    t_grid = torch.as_tensor(TENORS_DAYS, dtype=surf.dtype, device=surf.device)
    m = torch.as_tensor(moneyness, dtype=surf.dtype, device=surf.device)
    tau = torch.as_tensor(tau_days, dtype=surf.dtype, device=surf.device)
    tau = torch.clamp(tau, float(TENORS_DAYS[0]), float(TENORS_DAYS[-1]))
    m = torch.clamp(m, float(MONEYNESS[0]), float(MONEYNESS[-1]))

    # index along moneyness
    i = torch.searchsorted(m_grid, m.reshape(-1)).clamp(1, len(MONEYNESS) - 1)
    j = torch.searchsorted(t_grid, tau.reshape(-1)).clamp(1, len(TENORS_DAYS) - 1)
    # reshape helpers
    flat = surf.reshape(-1, 9, 6)
    n = flat.shape[0]
    i0, i1 = i - 1, i
    j0, j1 = j - 1, j
    m0, m1 = m_grid[i0], m_grid[i1]
    t0, t1 = t_grid[j0], t_grid[j1]
    wm = ((m.reshape(-1) - m0) / (m1 - m0 + 1e-12)).clamp(0, 1)
    wt = ((tau.reshape(-1) - t0) / (t1 - t0 + 1e-12)).clamp(0, 1)
    # gather — assume broadcast n matches
    if i0.numel() == 1:
        i0 = i0.expand(n)
        i1 = i1.expand(n)
        j0 = j0.expand(n)
        j1 = j1.expand(n)
        wm = wm.expand(n)
        wt = wt.expand(n)
    idx = torch.arange(n, device=surf.device)
    v00 = flat[idx, i0, j0]
    v01 = flat[idx, i0, j1]
    v10 = flat[idx, i1, j0]
    v11 = flat[idx, i1, j1]
    v0 = v00 * (1 - wt) + v01 * wt
    v1 = v10 * (1 - wt) + v11 * wt
    return (v0 * (1 - wm) + v1 * wm).reshape(surf.shape[:-2])


def _bs_call_put(S, K, T, r, sigma, kind: str) -> torch.Tensor:
    T = torch.clamp(T, min=1e-8)
    sigma = torch.clamp(sigma, min=1e-6)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    # torch erf-based CDF
    N = lambda x: 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    if kind == "call":
        return S * N(d1) - K * torch.exp(-r * T) * N(d2)
    return K * torch.exp(-r * T) * N(-d2) - S * N(-d1)


def vanilla_prices_along_path(
    S: torch.Tensor,
    surf: torch.Tensor,
    instruments: InstrumentSet,
    maturity_days: int,
    r: float = 0.065,
) -> torch.Tensor:
    """Prices [n, H+1, n_opt] for vanillas fixed at inception."""
    n, Hp1 = S.shape
    n_opt = len(instruments.kinds)
    S0 = S[:, 0]
    out = torch.zeros(n, Hp1, n_opt, dtype=S.dtype, device=S.device)
    for j in range(n_opt):
        K = instruments.strike_mults[j] * S0
        kind = instruments.kinds[j]
        tenor0 = instruments.tenor_days[j]
        for t in range(Hp1):
            rem = max(tenor0 - t, float(TENORS_DAYS[0]))
            mny = (K / S[:, t]).detach()  # moneyness vs current spot proxy for F
            # vectorized lookup per path
            ivs = []
            for i in range(n):
                ivs.append(lookup_iv(surf[i, t], float(mny[i]), rem))
            iv = torch.stack(ivs)
            T = rem / 365.0
            out[:, t, j] = _bs_call_put(S[:, t], K, T, r, iv, kind)
    return out


def instrument_set_for(label: str, barrier_mult: float = 0.9) -> InstrumentSet | None:
    if label == "I1":
        return None
    if label == "I2":
        return InstrumentSet(kinds=["call"], strike_mults=[1.0], tenor_days=[30])
    if label == "I3":
        return InstrumentSet(
            kinds=["call", "put"],
            strike_mults=[1.0, barrier_mult],
            tenor_days=[30, 30],
        )
    raise ValueError(label)
