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


def _to(paths: Paths, device) -> Paths:
    return Paths(S=paths.S.to(device).float(), surf=paths.surf.to(device).float(),
                 start_idx=paths.start_idx.to(device))


def _subset(paths: Paths, idx: torch.Tensor) -> Paths:
    return Paths(S=paths.S[idx], surf=paths.surf[idx], start_idx=paths.start_idx[idx])


def inception_price(paths: Paths, contract: Contract, r: float = 0.065) -> torch.Tensor:
    """Premium received at t=0: closed-form barrier price at the inception ATM 30d IV."""
    from niftyhedge.hedge.env import bs_barrier_price_torch

    S0 = paths.S[:, 0]
    iv = paths.surf[:, 0, 4, 2]
    K = contract.strike_mult * S0
    B = contract.barrier_mult * S0
    T = torch.full_like(S0, contract.maturity_days / 365.0)
    return bs_barrier_price_torch(S0, K, B, T, r, iv, kind=contract.kind).detach()


def evaluate_policy(policy, paths: Paths, contract: Contract, instruments, cost, alpha=0.95,
                    O: torch.Tensor | None = None, batch: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Terminal losses (normalised by S0) and hedge ratios for a trained policy."""
    from niftyhedge.hedge.instruments import vanilla_prices_along_path

    Ls, hs = [], []
    policy.eval()
    with torch.no_grad():
        for i in range(0, paths.S.shape[0], batch):
            sl = slice(i, i + batch)
            p = Paths(S=paths.S[sl], surf=paths.surf[sl], start_idx=paths.start_idx[sl])
            Ob = None
            if instruments is not None:
                Ob = O[sl] if O is not None else vanilla_prices_along_path(
                    p.S, p.surf, instruments, contract.maturity_days)
            h, q = rollout_policy(policy, p, contract)
            L = terminal_loss(p, contract, h, q, instruments, cost, inception_price(p, contract), O=Ob)
            Ls.append(L / p.S[:, 0])
            hs.append(h)
    return torch.cat(Ls), torch.cat(hs)


def train_deep_hedger(
    paths_train: Paths,
    paths_val: Paths,
    contract: Contract,
    instruments: InstrumentSet | None,
    cost: CostModel,
    *,
    seed: int = 0,
    epochs: int = 50,
    steps: int | None = None,
    batch_size: int = 4096,
    lr: float = 1e-3,
    alpha: float = 0.95,
    hidden: int = 128,
    residual: bool = True,
    eval_every: int = 50,
    out_dir: str | Path | None = None,
    smoke: bool = False,
    device: str | None = None,
) -> dict:
    """Pathwise deep hedging, mini-batch Adam on the Rockafellar-Uryasev CVaR.

    Losses are normalised by S0 (fraction of notional) so paths from 2018
    (NIFTY ~10k) and 2025 (~25k) carry equal weight. `steps` gradient steps
    (default: epochs * n_train / batch_size). The previous version took one
    full-batch step per epoch, i.e. only 50 updates in total.
    """
    from niftyhedge.hedge.baselines import bs_barrier_delta_bgk
    from niftyhedge.hedge.instruments import vanilla_prices_along_path

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths_train, paths_val = _to(paths_train, device), _to(paths_val, device)
    n_tr = paths_train.S.shape[0]
    batch_size = min(batch_size, n_tr)
    if steps is None:
        steps = max(1, epochs * n_tr // batch_size)
    if smoke:
        steps, eval_every = min(steps, 10), 5
        hidden = min(hidden, 64)
    n_opt = 0 if instruments is None else len(instruments.kinds)
    policy = HedgePolicy(state_dim=8, n_opt=n_opt, hidden=hidden, residual=residual).to(device)
    w = nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.Adam(list(policy.parameters()) + [w], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    O_tr = O_va = None
    if instruments is not None:  # marks are policy-independent: compute once
        O_tr = vanilla_prices_along_path(paths_train.S, paths_train.surf, instruments, contract.maturity_days)
        O_va = vanilla_prices_along_path(paths_val.S, paths_val.surf, instruments, contract.maturity_days)
    p0_tr = inception_price(paths_train, contract)

    # classical benchmark on the same validation paths (Gate 3 compares against this)
    hb, qb = bs_barrier_delta_bgk(paths_val, contract, instruments)
    Lb = terminal_loss(paths_val, contract, hb, qb, instruments, cost,
                       inception_price(paths_val, contract), O=O_va) / paths_val.S[:, 0]
    bs_val_cvar = float(cvar(Lb, alpha))

    def _val():
        Lv, hv = evaluate_policy(policy, paths_val, contract, instruments, cost, alpha, O=O_va)
        return float(cvar(Lv, alpha)), float(hv.std()), float((hv - hb).abs().mean())

    best, best_state, log = float("inf"), None, []
    cv, act_std, dev = _val()
    log.append({"step": 0, "train_obj": float("nan"), "val_cvar": cv, "action_std": act_std, "mean_abs_dev_from_bs": dev})
    best, best_state = cv, {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    for step in range(1, steps + 1):
        policy.train()
        idx = torch.randint(0, n_tr, (batch_size,), device=device)
        pb = _subset(paths_train, idx)
        h, q = rollout_policy(policy, pb, contract)
        L = terminal_loss(pb, contract, h, q, instruments, cost, p0_tr[idx],
                          O=None if O_tr is None else O_tr[idx]) / pb.S[:, 0]
        obj = ru_objective(L, w, alpha)
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            cv, act_std, dev = _val()
            log.append({"step": step, "train_obj": float(obj.detach()), "val_cvar": cv,
                        "action_std": act_std, "mean_abs_dev_from_bs": dev})
            if cv < best:
                best = cv
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    policy.load_state_dict(best_state)
    meta = {"state_dim": 8, "n_opt": n_opt, "hidden": hidden, "residual": residual,
            "contract": contract.__dict__, "val_cvar": best, "bs_val_cvar": bs_val_cvar,
            "steps": steps, "batch_size": batch_size, "n_train": n_tr, "loss_units": "fraction_of_S0"}
    result = {"policy": policy, "val_cvar": best, "bs_val_cvar": bs_val_cvar, "log": log,
              "action_std": log[-1]["action_std"], "meta": meta}
    if out_dir is not None:
        import csv, json

        out_dir = ensure_dir(out_dir)
        torch.save(policy.state_dict(), out_dir / "policy.pt")
        (out_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        with open(out_dir / "train_log.csv", "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(log[0]))
            wri.writeheader()
            wri.writerows(log)
    return result


def load_policy(run_dir: str | Path, device: str = "cpu") -> HedgePolicy:
    import json

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "policy_meta.json").read_text())
    pol = HedgePolicy(state_dim=meta["state_dim"], n_opt=meta["n_opt"], hidden=meta["hidden"],
                      residual=meta.get("residual", False))
    pol.load_state_dict(torch.load(run_dir / "policy.pt", map_location=device))
    return pol.to(device).eval()


def policy_hedger(policy: HedgePolicy):
    """Wrap a trained policy as a baseline-style hedger(paths, contract, instruments)."""

    def hedger(paths: Paths, contract: Contract, instruments=None):
        with torch.no_grad():
            return rollout_policy(policy, paths, contract)

    return hedger
