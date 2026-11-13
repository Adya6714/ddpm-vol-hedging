"""Classical hedging baselines. Each returns (h, q) using info up to t only."""

from __future__ import annotations

import torch

from niftyhedge.gen.base import Paths
from niftyhedge.hedge.contracts import Contract
from niftyhedge.hedge.env import bs_barrier_delta
from niftyhedge.hedge.instruments import InstrumentSet


def no_hedge(paths: Paths, contract: Contract, instruments: InstrumentSet | None = None):
    n, Hp1 = paths.S.shape
    H = Hp1 - 1
    h = torch.zeros(n, H, dtype=paths.S.dtype, device=paths.S.device)
    q = None
    if instruments is not None:
        q = torch.zeros(n, H, len(instruments.kinds), dtype=paths.S.dtype, device=paths.S.device)
    return h, q


def _delta_path(
    paths: Paths,
    contract: Contract,
    *,
    bgk: bool = False,
    smile: bool = False,
    sign: float = 1.0,
    r: float = 0.065,
) -> torch.Tensor:
    S = paths.S
    n, Hp1 = S.shape
    H = Hp1 - 1
    S0 = S[:, 0]
    K = contract.strike_mult * S0
    B = contract.barrier_mult * S0
    h = torch.zeros(n, H, dtype=S.dtype, device=S.device)
    for t in range(H):
        rem = max(contract.maturity_days - t, 1) / 365.0
        T = torch.full((n,), rem, dtype=S.dtype, device=S.device)
        # ATM IV from surface cell [4,2]; smile uses barrier moneyness column
        if smile:
            # nearest moneyness to B/S
            mny = (B / S[:, t]).cpu()
            # use column closest to barrier strike IV at 30d (index 2)
            iv = paths.surf[:, t, 3, 2]  # ~0.95 moneyness as proxy
        else:
            iv = paths.surf[:, t, 4, 2]
        d = bs_barrier_delta(
            S[:, t], K, B, T, r, iv, kind=contract.kind, bgk=bgk
        )
        h[:, t] = sign * d
    return h


def bs_barrier_delta_hedge(paths: Paths, contract: Contract, instruments=None, **kw):
    h = _delta_path(paths, contract, bgk=False, sign=1.0)
    q = None
    if instruments is not None:
        q = torch.zeros(paths.S.shape[0], paths.S.shape[1] - 1, len(instruments.kinds), device=paths.S.device)
    return h, q


def bs_barrier_delta_bgk(paths: Paths, contract: Contract, instruments=None, **kw):
    h = _delta_path(paths, contract, bgk=True, sign=1.0)
    q = None
    if instruments is not None:
        q = torch.zeros(paths.S.shape[0], paths.S.shape[1] - 1, len(instruments.kinds), device=paths.S.device)
    return h, q


def smile_delta(paths: Paths, contract: Contract, instruments=None, **kw):
    h = _delta_path(paths, contract, bgk=True, smile=True, sign=1.0)
    q = None
    if instruments is not None:
        q = torch.zeros(paths.S.shape[0], paths.S.shape[1] - 1, len(instruments.kinds), device=paths.S.device)
    return h, q


def ww_band(paths: Paths, contract: Contract, instruments=None, band: float = 0.05, **kw):
    """Whalley–Wilmott style band around BGK delta."""
    h_star, q = bs_barrier_delta_bgk(paths, contract, instruments)
    h = torch.zeros_like(h_star)
    prev = torch.zeros(h_star.shape[0], device=h_star.device, dtype=h_star.dtype)
    for t in range(h_star.shape[1]):
        target = h_star[:, t]
        move = (target - prev).abs() > band
        prev = torch.where(move, target, prev)
        h[:, t] = prev
    return h, q


def delta_vega(paths: Paths, contract: Contract, instruments: InstrumentSet | None = None, **kw):
    """I2: underlying delta + unit ATM call to offset vega (simple static ratio)."""
    h, _ = bs_barrier_delta_bgk(paths, contract, None)
    if instruments is None:
        return h, None
    n, H = h.shape
    q = torch.ones(n, H, len(instruments.kinds), device=h.device, dtype=h.dtype) * 0.1
    return h, q


def static_replication(paths: Paths, contract: Contract, instruments: InstrumentSet | None = None, **kw):
    """I3 placeholder: hold barrier-strike put statically + residual BGK delta."""
    h, _ = bs_barrier_delta_bgk(paths, contract, None)
    if instruments is None:
        return h, None
    n, H = h.shape
    q = torch.zeros(n, H, len(instruments.kinds), device=h.device, dtype=h.dtype)
    if len(instruments.kinds) >= 2:
        q[:, :, 1] = 1.0  # long put at barrier
    return h, q


BASELINES = {
    "no_hedge": no_hedge,
    "bs_barrier_delta": bs_barrier_delta_hedge,
    "bs_barrier_delta_bgk": bs_barrier_delta_bgk,
    "smile_delta": smile_delta,
    "ww_band": ww_band,
    "delta_vega": delta_vega,
    "static_replication": static_replication,
}
