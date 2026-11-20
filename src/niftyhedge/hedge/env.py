"""Vectorised terminal-loss hedging environment."""

from __future__ import annotations

import math

import torch

from niftyhedge.gen.base import Paths
from niftyhedge.hedge.contracts import Contract, barrier_payoff, knock_mask
from niftyhedge.hedge.instruments import CostModel, InstrumentSet, vanilla_prices_along_path
from niftyhedge.utils.io import assert_finite


def terminal_loss(
    paths: Paths,
    contract: Contract,
    h: torch.Tensor,
    q: torch.Tensor | None,
    instruments: InstrumentSet | None,
    cost: CostModel,
    p0: torch.Tensor,
    r: float = 0.065,
    O: torch.Tensor | None = None,
) -> torch.Tensor:
    """L = payoff - p0 - sum h*dS - sum q*dO + costs. Short one option.

    paths.S: [n, H+1], h: [n, H], q: [n, H, n_opt] or None.
    Positive loss is bad for the hedger.
    """
    S = paths.S
    assert_finite(S, "S")
    n, Hp1 = S.shape
    H = Hp1 - 1
    if h.shape != (n, H):
        raise ValueError(f"h shape {h.shape} != {(n, H)}")

    payoff = barrier_payoff(S, contract)
    dS = S[:, 1:] - S[:, :-1]
    # stop accruing hedge gains after knock: zero h after knock day
    knocked = knock_mask(S, contract)  # [n, H+1]
    alive = ~knocked[:, :-1]  # held over (t,t+1] only if not yet knocked at t
    h_eff = h * alive.float()
    hedge_pnl = (h_eff * dS).sum(dim=1)

    opt_pnl = torch.zeros(n, dtype=S.dtype, device=S.device)
    opt_cost = torch.zeros(n, dtype=S.dtype, device=S.device)
    if q is not None and instruments is not None:
        if O is None:  # option marks do not depend on the policy: pass them precomputed
            O = vanilla_prices_along_path(S, paths.surf, instruments, contract.maturity_days, r=r)
        assert_finite(O, "O")
        dO = O[:, 1:] - O[:, :-1]
        q_eff = q * alive.float().unsqueeze(-1)
        opt_pnl = (q_eff * dO).sum(dim=(1, 2))
        # costs on option rebalancing
        q0 = torch.zeros(n, 1, q.shape[-1], dtype=q.dtype, device=q.device)
        dq = torch.cat([q[:, :1] - q0[:, :1], q[:, 1:] - q[:, :-1]], dim=1)
        prem = O[:, :-1].abs()
        opt_cost = (cost.option_frac * dq.abs() * prem).sum(dim=(1, 2))

    # underlying costs
    h0 = torch.zeros(n, 1, dtype=h.dtype, device=h.device)
    dh = torch.cat([h[:, :1], h[:, 1:] - h[:, :-1]], dim=1)
    und_cost = (cost.underlying_bps * 1e-4 * dh.abs() * S[:, :-1]).sum(dim=1)

    # short option: we receive p0, pay payoff, gain from hedges
    L = payoff - p0 - hedge_pnl - opt_pnl + und_cost + opt_cost
    assert_finite(L, "terminal_loss")
    return L


def bs_barrier_price_torch(
    S: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    T: torch.Tensor,
    r: float,
    sigma: torch.Tensor,
    kind: str = "DOC",
    q: float = 0.0,
) -> torch.Tensor:
    """Continuous-monitoring barrier calls (Merton / Reiner-Rubinstein; Hull ch. 26).

    DOC: down-and-out call, valid for B <= K (the usual case, B = 0.9..0.97 S0).
    UOC: up-and-out call, B > K.
    Previous version scaled the reflected term by (B/S)^(2*lam) instead of
    (B/S)^(2*lam-2) and returned the *vanilla* price for UOC.
    """
    S = torch.as_tensor(S)
    K = torch.as_tensor(K, dtype=S.dtype, device=S.device)
    B = torch.as_tensor(B, dtype=S.dtype, device=S.device)
    T = torch.as_tensor(T, dtype=S.dtype, device=S.device)
    sigma = torch.clamp(torch.as_tensor(sigma, dtype=S.dtype, device=S.device), min=1e-6)
    T = torch.clamp(T, min=1e-8)
    sqT = torch.sqrt(T)
    sT = sigma * sqT
    N = lambda x: 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    dq, dr = torch.exp(-q * T), torch.exp(-r * T)
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    vanilla = S * dq * N(d1) - K * dr * N(d2)
    if kind == "VANILLA_CALL":
        return vanilla
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = torch.log(B * B / (S * K)) / sT + lam * sT
    if kind == "DOC":
        c_di = S * dq * (B / S) ** (2 * lam) * N(y) - K * dr * (B / S) ** (2 * lam - 2) * N(y - sT)
        price = torch.clamp(vanilla - c_di, min=0.0)
        return torch.where(S <= B, torch.zeros_like(price), price)
    if kind == "UOC":
        x1 = torch.log(S / B) / sT + lam * sT
        y1 = torch.log(B / S) / sT + lam * sT
        c_ui = (
            S * dq * N(x1)
            - K * dr * N(x1 - sT)
            - S * dq * (B / S) ** (2 * lam) * (N(-y) - N(-y1))
            + K * dr * (B / S) ** (2 * lam - 2) * (N(-y + sT) - N(-y1 + sT))
        )
        price = torch.clamp(vanilla - c_ui, min=0.0)
        return torch.where(S >= B, torch.zeros_like(price), price)
    raise ValueError(f"unknown barrier kind {kind}")


def bs_barrier_delta(
    S: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    T: torch.Tensor,
    r: float,
    sigma: torch.Tensor,
    kind: str = "DOC",
    bgk: bool = False,
    dt: float = 1.0 / 252.0,
) -> torch.Tensor:
    """Delta of closed-form barrier price (central finite difference; works under no_grad)."""
    if bgk and kind == "DOC":
        B = B * torch.exp(-0.5826 * sigma * math.sqrt(dt))
    elif bgk and kind == "UOC":
        B = B * torch.exp(0.5826 * sigma * math.sqrt(dt))
    eps = (0.001 * S).clamp(min=1e-4)
    up = bs_barrier_price_torch(S + eps, K, B, T, r, sigma, kind=kind)
    dn = bs_barrier_price_torch(S - eps, K, B, T, r, sigma, kind=kind)
    return ((up - dn) / (2 * eps)).detach().clamp(-2.0, 2.0)
