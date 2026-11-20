"""Real PPO ablation on the same vectorised hedging environment.

The earlier file called `train_deep_hedger` and labelled the result "PPO", so
the PPO-vs-deep-hedging ablation compared deep hedging with itself.

This is a genuine on-policy PPO:
  * Gaussian actor over the hedge ratio, same HedgePolicy backbone (residual on
    the BGK delta, so it starts from the classical hedge like the deep hedger);
  * learned value baseline on the same state;
  * episodic reward = -(L / S0)^2 * 1e4 at expiry (variance-style objective as in
    the original paper), zero before; gamma = 1, so the return-to-go is the
    terminal reward and the advantage is (R - V(s_t));
  * clipped surrogate, several epochs per batch of sampled episodes.
Evaluation uses the deterministic mean action, scored with the same CVaR as the
deep hedger, so the ablation isolates the *training algorithm*.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from niftyhedge.gen.base import Paths
from niftyhedge.hedge.contracts import Contract
from niftyhedge.hedge.deep_hedge import _subset, _to, evaluate_policy, inception_price
from niftyhedge.hedge.env import bs_barrier_delta, terminal_loss
from niftyhedge.hedge.instruments import CostModel
from niftyhedge.hedge.policy import HedgePolicy, build_state
from niftyhedge.hedge.risk import cvar
from niftyhedge.utils.io import ensure_dir
from niftyhedge.utils.seed import set_seed


class PPOActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.policy = HedgePolicy(state_dim, n_opt=0, hidden=hidden, residual=True)
        self.log_std = nn.Parameter(torch.full((1,), -2.0))  # initial std ~0.14 delta units
        self.value = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                   nn.SiLU(), nn.Linear(hidden, 1))

    def dist(self, state):
        mu, _ = self.policy(state)
        return Normal(mu, self.log_std.exp().expand_as(mu))


def _episode(actor: PPOActor, paths: Paths, contract: Contract, r: float = 0.065, sample: bool = True):
    S, surf = paths.S, paths.surf
    n, Hp1 = S.shape
    H = Hp1 - 1
    K = contract.strike_mult * S[:, 0]
    B = contract.barrier_mult * S[:, 0]
    h = torch.zeros(n, H, device=S.device)
    states, logps = [], []
    h_prev = torch.zeros(n, device=S.device)
    for t in range(H):
        T = torch.full((n,), max(contract.maturity_days - t, 1) / 365.0, device=S.device)
        bs_d = bs_barrier_delta(S[:, t], K, B, T, r, surf[:, t, 4, 2], kind=contract.kind, bgk=True)
        st = build_state(S, surf, t, contract.maturity_days, K, B, h_prev, bs_d)
        d = actor.dist(st)
        a = d.sample() if sample else d.mean
        states.append(st)
        logps.append(d.log_prob(a))
        h[:, t] = a
        h_prev = a
    return h, torch.stack(states, 1), torch.stack(logps, 1)


def train_ppo(
    paths_train: Paths,
    paths_val: Paths,
    contract: Contract,
    cost: CostModel,
    *,
    seed: int = 0,
    epochs: int = 20,
    iterations: int = 300,
    batch_size: int = 4096,
    ppo_epochs: int = 4,
    clip: float = 0.2,
    lr: float = 3e-4,
    hidden: int = 128,
    alpha: float = 0.95,
    out_dir: str | Path | None = None,
    smoke: bool = False,
    device: str | None = None,
) -> dict:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if smoke:
        iterations, hidden = 3, 32
    paths_train, paths_val = _to(paths_train, device), _to(paths_val, device)
    n_tr = paths_train.S.shape[0]
    batch_size = min(batch_size, n_tr)
    actor = PPOActor(8, hidden).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    p0_tr = inception_price(paths_train, contract)
    log, best, best_state = [], float("inf"), None
    for it in range(iterations):
        idx = torch.randint(0, n_tr, (batch_size,), device=device)
        pb = _subset(paths_train, idx)
        with torch.no_grad():
            h, states, logp_old = _episode(actor, pb, contract)
            L = terminal_loss(pb, contract, h, None, None, cost, p0_tr[idx]) / pb.S[:, 0]
            R = -(L**2) * 1e4  # terminal reward, gamma = 1
        Rt = R[:, None].expand_as(logp_old)
        for _ in range(ppo_epochs):
            d = actor.dist(states)
            logp = d.log_prob(h)
            V = actor.value(states).squeeze(-1)
            adv = (Rt - V).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = (logp - logp_old).exp()
            pol_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            v_loss = ((V - Rt) ** 2).mean()
            loss = pol_loss + 0.5 * v_loss - 1e-3 * d.entropy().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
        if (it + 1) % max(1, iterations // 30) == 0 or it == iterations - 1:
            Lv, hv = evaluate_policy(actor.policy, paths_val, contract, None, cost, alpha)
            cv = float(cvar(Lv, alpha))
            log.append({"iter": it + 1, "mean_R": float(R.mean()), "val_cvar": cv,
                        "action_std": float(hv.std()), "policy_std": float(actor.log_std.exp())})
            if cv < best:
                best = cv
                best_state = {k: v.detach().cpu().clone() for k, v in actor.policy.state_dict().items()}
    if best_state is not None:
        actor.policy.load_state_dict(best_state)
    meta = {"algo": "ppo", "state_dim": 8, "n_opt": 0, "hidden": hidden, "residual": True,
            "contract": contract.__dict__, "val_cvar": best, "iterations": iterations,
            "loss_units": "fraction_of_S0"}
    result = {"policy": actor.policy, "val_cvar": best, "log": log,
              "action_std": log[-1]["action_std"] if log else 0.0, "meta": meta}
    if out_dir is not None:
        import csv, json

        out_dir = ensure_dir(out_dir)
        torch.save(actor.policy.state_dict(), out_dir / "policy.pt")
        (out_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        if log:
            with open(out_dir / "train_log.csv", "w", newline="") as f:
                wri = csv.DictWriter(f, fieldnames=list(log[0]))
                wri.writeheader()
                wri.writerows(log)
    return result
