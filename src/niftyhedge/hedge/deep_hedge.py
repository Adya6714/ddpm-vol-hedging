"""Pathwise deep hedging with Rockafellar–Uryasev CVaR."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from niftyhedge.gen.base import Paths
from niftyhedge.hedge.contracts import Contract
from niftyhedge.hedge.env import bs_barrier_delta, terminal_loss
from niftyhedge.hedge.instruments import CostModel, InstrumentSet
from niftyhedge.hedge.policy import HedgePolicy, build_state
from niftyhedge.hedge.risk import cvar, ru_objective
from niftyhedge.utils.io import ensure_dir, write_manifest
from niftyhedge.utils.seed import set_seed


def rollout_policy(
    policy: HedgePolicy,
    paths: Paths,
    contract: Contract,
    *,
    use_bs_delta: bool = True,
    r: float = 0.065,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Produce h [n,H] and optional q [n,H,n_opt] causally."""
    S, surf = paths.S, paths.surf
    n, Hp1 = S.shape
    H = Hp1 - 1
    S0 = S[:, 0]
    K = contract.strike_mult * S0
    B = contract.barrier_mult * S0
    h = torch.zeros(n, H, device=S.device, dtype=S.dtype)
    q = None
    if policy.n_opt > 0:
        q = torch.zeros(n, H, policy.n_opt, device=S.device, dtype=S.dtype)
    h_prev = torch.zeros(n, device=S.device, dtype=S.dtype)
    for t in range(H):
        bs_d = None
        if use_bs_delta:
            rem = max(contract.maturity_days - t, 1) / 365.0
            T = torch.full((n,), rem, device=S.device, dtype=S.dtype)
            iv = surf[:, t, 4, 2]
            bs_d = bs_barrier_delta(S[:, t], K, B, T, r, iv, kind=contract.kind, bgk=True)
        state = build_state(S, surf, t, contract.maturity_days, K, B, h_prev, bs_d)
        ht, qt = policy(state)
        h[:, t] = ht
        if q is not None and qt is not None:
            q[:, t] = qt
        h_prev = ht.detach()
    return h, q


def train_deep_hedger(
    paths_train: Paths,
    paths_val: Paths,
    contract: Contract,
    instruments: InstrumentSet | None,
    cost: CostModel,
    *,
    seed: int = 0,
    epochs: int = 50,
    lr: float = 1e-3,
    alpha: float = 0.95,
    out_dir: str | Path | None = None,
    smoke: bool = False,
) -> dict:
    set_seed(seed)
    if smoke:
        epochs = min(epochs, 5)
    n_opt = 0 if instruments is None else len(instruments.kinds)
    state_dim = 8  # with bs delta
    policy = HedgePolicy(state_dim=state_dim, n_opt=n_opt, hidden=64 if smoke else 128)
    w = nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam(list(policy.parameters()) + [w], lr=lr)

    # p0 from inception ATM BGK-ish price proxy
    def _p0(paths: Paths) -> torch.Tensor:
        S0 = paths.S[:, 0]
        iv = paths.surf[:, 0, 4, 2]
        # crude BS vanilla * 0.7 as barrier proxy for training scale
        from niftyhedge.hedge.env import bs_barrier_price_torch

        K = contract.strike_mult * S0
        B = contract.barrier_mult * S0
        T = torch.full_like(S0, contract.maturity_days / 365.0)
        return bs_barrier_price_torch(S0, K, B, T, 0.065, iv, kind=contract.kind).detach()

    p0_tr = _p0(paths_train)
    p0_va = _p0(paths_val)
    best = float("inf")
    best_state = None
    log = []
    for ep in range(epochs):
        policy.train()
        h, q = rollout_policy(policy, paths_train, contract)
        L = terminal_loss(paths_train, contract, h, q, instruments, cost, p0_tr)
        obj = ru_objective(L, w, alpha)
        opt.zero_grad()
        obj.backward()
        opt.step()

        policy.eval()
        with torch.no_grad():
            hv, qv = rollout_policy(policy, paths_val, contract)
            Lv = terminal_loss(paths_val, contract, hv, qv, instruments, cost, p0_va)
            cv = float(cvar(Lv, alpha))
            act_std = float(hv.std())
        log.append({"epoch": ep, "train_obj": float(obj.detach()), "val_cvar": cv, "action_std": act_std})
        if cv < best:
            best = cv
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)

    result = {"policy": policy, "val_cvar": best, "log": log, "action_std": log[-1]["action_std"] if log else 0.0}
    if out_dir is not None:
        out_dir = ensure_dir(out_dir)
        torch.save(policy.state_dict(), out_dir / "policy.pt")
        write_manifest(out_dir, config={"contract": contract.__dict__, "seed": seed}, seed=seed, extra={"val_cvar": best})
        import csv

        with open(out_dir / "train_log.csv", "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=["epoch", "train_obj", "val_cvar", "action_std"])
            wri.writeheader()
            wri.writerows(log)
    return result
