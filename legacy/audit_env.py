"""Audit of the paper's evaluation, reusing the notebook's V2 env + metric verbatim.

Compares: BS +delta (paper baseline), zero hedge, BS -delta, saved agents.
Common random numbers across strategies (same seed per episode).

Usage (from repo root, CPU):
  python legacy/audit_env.py . 300
  python legacy/audit_env.py /path/to/repo 300 | tee runs/legacy_audit.txt

Expected Normal-regime ordering (2026-11-13 audit, less negative = better):
  BS +delta ≈ -0.0209  <  No hedge ≈ -0.0122  <  DDPM-RL ≈ -0.0087  <  BS -delta ≈ -0.0078
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as N

if len(sys.argv) < 3:
    print(__doc__)
    print("Usage: python legacy/audit_env.py <repo_root> <n_episodes>")
    sys.exit(1)

R = Path(sys.argv[1]).resolve()
OPTION_MATURITY = 30
RISK_FREE = 0.065
TC_RATE = 0.001
STATE_DIM = 15


def price(S, K, B, T, r, s):
    if S <= B:
        return 0.0
    if T <= 1e-6:
        return max(S - K, 0.0)
    d1 = lambda x: (math.log(x / K) + (r + 0.5 * s * s) * T) / (s * math.sqrt(T))
    d2 = lambda x: d1(x) - s * math.sqrt(T)
    van = S * N.cdf(d1(S)) - K * math.exp(-r * T) * N.cdf(d2(S))
    lam = (r + 0.5 * s * s) / s**2
    bt = (B / S) ** (2 * lam) * (
        B * B / S * N.cdf(d1(B * B / S)) - K * math.exp(-r * T) * N.cdf(d2(B * B / S))
    )
    return max(van - bt, 0.0)


def delta(S, K, B, T, r, s):
    if S <= B:
        return 0.0
    dS = S * 0.001
    return float(
        np.clip(
            (price(S + dS, K, B, T, r, s) - price(S - dS, K, B, T, r, s)) / (2 * dS),
            -2,
            2,
        )
    )


class Env:  # NiftyBarrierEnvV2 from restore cell, verbatim logic
    def __init__(s, lib):
        s.surfaces = lib
        s.T_days = 30
        s.r = RISK_FREE
        s.tc = TC_RATE

    def vol(s):
        v = s.surfaces[min(s.i, len(s.surfaces) - 1), 4, 2]
        return float(v) if not np.isnan(v) and v > 0 else 0.16

    def feats(s):
        q = s.surfaces[min(s.i, len(s.surfaces) - 1)]
        g = lambda a, b: (float(q[a, b]) if not np.isnan(q[a, b]) else 0.16)
        return np.array(
            [
                g(4, 0),
                g(4, 2),
                g(4, 4),
                g(3, 2) - g(4, 2),
                g(3, 4) - g(4, 4),
                g(4, 4) - g(4, 2),
                0.5 * (g(2, 2) + g(6, 2)) - g(4, 2),
                g(2, 2),
                g(6, 2),
                float(np.nanmean(q)),
            ],
            dtype=np.float32,
        )

    def state(s, ret):
        base = np.array(
            [
                ret * 10,
                math.log(s.S / s.K) * 5,
                (s.T_days - s.t) / s.T_days,
                s.hedge,
                (s.S - s.B) / s.B * 5,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, (s.feats() - 0.15) / 0.08])

    def reset(s):
        s.i = np.random.randint(0, max(1, len(s.surfaces) - s.T_days - 1))
        s.S = 20000.0 * np.random.uniform(0.95, 1.05)
        s.K = s.S
        s.B = 0.9 * s.S
        s.t = 0
        s.hedge = 0.0
        s.knocked = False
        s.V = price(s.S, s.K, s.B, s.T_days / 252.0, s.r, s.vol())
        return s.state(0.0)

    def step(s, a):
        a = float(np.clip(a, -2, 2))
        s.t += 1
        sig = s.vol()
        dt = 1 / 252.0
        lr = (s.r - 0.5 * sig * sig) * dt + sig * math.sqrt(dt) * np.random.standard_normal()
        Sn = s.S * math.exp(lr)
        if Sn <= s.B:
            s.knocked = True
            he = -s.V + s.hedge * (Sn - s.S)
            s.S = Sn
            return s.state(lr), True, he
        s.i = min(s.i + 1, len(s.surfaces) - 1)
        Tn = (s.T_days - s.t) / 252.0
        Vn = max(Sn - s.K, 0.0) if Tn <= 0 else price(Sn, s.K, s.B, Tn, s.r, s.vol())
        dV = Vn - s.V
        pnl = dV + s.hedge * (Sn - s.S) - s.tc * abs(a - s.hedge) * Sn
        s.S = Sn
        s.V = Vn
        s.hedge = a
        return s.state(lr), s.t >= s.T_days, pnl


class RB(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d))
        s.norm = nn.LayerNorm(d)

    def forward(s, x):
        return s.norm(x + s.net(x))


class Agent(nn.Module):
    def __init__(s, h=256):
        super().__init__()
        s.backbone = nn.Sequential(nn.Linear(STATE_DIM, h), nn.SiLU(), RB(h), RB(h), RB(h))
        s.actor_mean = nn.Linear(h, 1)
        s.actor_log_std = nn.Parameter(torch.zeros(1))
        s.critic = nn.Linear(h, 1)

    def act(s, x):
        with torch.no_grad():
            return float(torch.tanh(s.actor_mean(s.backbone(torch.tensor(x)[None]))) * 2.0)


def load(p):
    a = Agent()
    a.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    a.eval()
    return a


def run(policy, lib, n, seed0=0):
    """returns per-episode paper metric (mean of worst-5% daily pnl/S), plus actions & deltas"""
    out = []
    acts = []
    dels = []
    for e in range(n):
        np.random.seed(seed0 + e)
        env = Env(lib)
        st = env.reset()
        done = False
        pn = []
        while not done:
            d = delta(env.S, env.K, env.B, (30 - env.t) / 252.0, env.r, env.vol())
            a = policy(st, d)
            acts.append(a)
            dels.append(d)
            pS = env.S
            st, done, p = env.step(a)
            pn.append(p / pS)
        pn = np.array(pn)
        v = np.quantile(pn, 0.05)
        out.append(pn[pn <= v].mean())
    return np.array(out), np.array(acts), np.array(dels)


def main():
    required = [
        R / "models/rl/best_agent_v2.pt",
        R / "models/rl/best_agent_heston.pt",
        R / "models/rl/best_agent.pt",
        R / "data/scenarios/scenarios_normal.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required audit artefacts:\n  " + "\n  ".join(missing))

    ag = load(R / "models/rl/best_agent_v2.pt")
    ah = load(R / "models/rl/best_agent_heston.pt")
    a1 = load(R / "models/rl/best_agent.pt")
    torch.manual_seed(0)
    fresh = Agent()
    nn.init.orthogonal_(fresh.actor_mean.weight, gain=0.01)
    nn.init.zeros_(fresh.actor_mean.bias)

    pols = {
        "BS +delta (paper baseline)": lambda s, d: d,
        "No hedge (h=0)": lambda s, d: 0.0,
        "BS -delta (sign-corrected)": lambda s, d: -d,
        "DDPM-RL (best_agent_v2)": lambda s, d: ag.act(s),
        "Heston-RL (best_agent_heston)": lambda s, d: ah.act(s),
        "V1 agent (best_agent)": lambda s, d: a1.act(s),
        "Fresh untrained agent": lambda s, d: fresh.act(s),
    }
    n = int(sys.argv[2])
    print(f"repo={R}  episodes={n}  (legacy paper metric; NOT terminal CVaR)")
    for reg in ["normal", "stressed", "event"]:
        lib = np.load(R / f"data/scenarios/scenarios_{reg}.npy")
        print(f"\n== {reg} (n={n}, common random numbers) ==")
        for k, p in pols.items():
            m, a, d = run(p, lib, n)
            c = np.corrcoef(a, d)[0, 1] if a.std() > 1e-9 else float("nan")
            print(
                f"  {k:32s} metric={m.mean():+.5f}  mean|action|={np.abs(a).mean():.3f} "
                f"mean action={a.mean():+.3f} corr(action,delta)={c:+.2f}"
            )


if __name__ == "__main__":
    main()
