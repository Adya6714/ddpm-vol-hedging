"""Real-NIFTY evaluation: walk-forward windows + trained-hedger discovery.

This is the paper's main experiment: a hedger trained in simulated world W on
fold k is scored on REAL 30-day NIFTY windows (futures-implied spot + market
IV surfaces) that start in fold k's test year. Pooling folds 0..5 gives a
walk-forward test over 2020 - Mar 2026 with no look-ahead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from niftyhedge.gen.base import Paths
from niftyhedge.hedge.baselines import BASELINES
from niftyhedge.hedge.contracts import Contract
from niftyhedge.hedge.deep_hedge import inception_price, load_policy, policy_hedger
from niftyhedge.hedge.env import terminal_loss
from niftyhedge.hedge.instruments import CostModel, InstrumentSet, vanilla_prices_along_path

CLASSICAL = ["no_hedge", "bs_barrier_delta", "bs_barrier_delta_bgk", "smile_delta", "ww_band"]


def load_real_market(surf_dir: str | Path = "data/surfaces"):
    surf_dir = Path(surf_dir)
    surfaces = np.load(surf_dir / "surfaces_v2.npy")
    dates = np.array([str(d) for d in np.load(surf_dir / "surface_dates_v2.npy", allow_pickle=True)])
    sp_path = surf_dir / "spot_v2.parquet"
    if not sp_path.exists():
        raise FileNotFoundError(f"{sp_path} missing - re-run scripts/01_build_surfaces.py")
    sp = pd.read_parquet(sp_path)
    if list(sp["date"].astype(str)) != list(dates):
        raise ValueError("spot_v2 / surface dates mismatch")
    return dates, sp["spot"].to_numpy(dtype=float), surfaces


def real_windows(dates, spot, surfaces, start_dates, horizon: int) -> tuple[Paths, np.ndarray]:
    """Overlapping real windows of `horizon` trading days starting on start_dates."""
    pos = {d: i for i, d in enumerate(dates)}
    starts = [pos[d] for d in map(str, start_dates) if d in pos and pos[d] + horizon < len(dates)]
    starts = np.asarray(starts, dtype=int)
    if len(starts) == 0:
        raise ValueError("no complete real windows for these start dates")
    ix = starts[:, None] + np.arange(horizon + 1)[None, :]
    S = torch.as_tensor(spot[ix], dtype=torch.float32)
    surf = torch.as_tensor(surfaces[ix], dtype=torch.float32)
    return Paths(S=S, surf=surf, start_idx=torch.as_tensor(starts)), np.asarray(dates)[starts]


def discover_runs(root: str | Path, contract_stem: str, instruments: str, fold: int) -> dict[str, list[Path]]:
    """{world: [seed run dirs]} for trained policies of this contract/instrument/fold."""
    out: dict[str, list[Path]] = {}
    for d in sorted(Path(root).glob(f"*/{instruments}/{contract_stem}/fold_{fold}/seed_*")):
        if (d / "policy.pt").exists() and (d / "policy_meta.json").exists():
            out.setdefault(d.parts[-5], []).append(d)
    return out


def losses_for(hedger: Callable, paths: Paths, contract: Contract, instruments: InstrumentSet | None,
               cost: CostModel, O: torch.Tensor | None = None) -> np.ndarray:
    """Terminal loss / S0 per window (premium = closed-form price at inception IV)."""
    h, q = hedger(paths, contract, instruments)
    if instruments is not None and q is None:
        q = torch.zeros(*h.shape, len(instruments.kinds), dtype=h.dtype, device=h.device)
    L = terminal_loss(paths, contract, h, q, instruments, cost, inception_price(paths, contract), O=O)
    return (L / paths.S[:, 0]).detach().cpu().numpy()


def evaluate_all(paths: Paths, contract: Contract, instruments: InstrumentSet | None, cost: CostModel,
                 runs: dict[str, list[Path]], ppo_runs: dict[str, list[Path]] | None = None,
                 classical: list[str] = CLASSICAL) -> dict[str, dict[str, np.ndarray]]:
    """{hedger_name: {seed_label: losses}}. Classical baselines have one 'seed'."""
    O = None
    if instruments is not None:
        O = vanilla_prices_along_path(paths.S, paths.surf, instruments, contract.maturity_days)
    res: dict[str, dict[str, np.ndarray]] = {}
    for name in classical:
        res[name] = {"-": losses_for(BASELINES[name], paths, contract, None, cost)}
    for tag, rr in [("deep", runs), ("ppo", ppo_runs or {})]:
        for world, dirs in rr.items():
            for d in dirs:
                pol = load_policy(d)
                res.setdefault(f"{tag}[{world}]", {})[d.name] = losses_for(
                    policy_hedger(pol), paths, contract, instruments if tag == "deep" else None, cost,
                    O=O if tag == "deep" else None)
    return res


def run_meta(d: Path) -> dict:
    return json.loads((Path(d) / "policy_meta.json").read_text())
