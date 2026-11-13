"""Real-market walk-forward backtest."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from niftyhedge.gen.base import Paths
from niftyhedge.hedge.contracts import Contract
from niftyhedge.hedge.env import terminal_loss
from niftyhedge.hedge.instruments import CostModel, InstrumentSet
from niftyhedge.eval.hedge_metrics import summarize_loss


def overlapping_windows(
    spot: np.ndarray,
    surfaces: np.ndarray,
    horizon: int = 30,
) -> Paths:
    """Build Paths from overlapping real windows."""
    n = len(spot) - horizon
    if n <= 0:
        raise ValueError("Not enough days for horizon")
    S = torch.zeros(n, horizon + 1)
    surf = torch.zeros(n, horizon + 1, 9, 6)
    for i in range(n):
        S[i] = torch.as_tensor(spot[i : i + horizon + 1], dtype=torch.float32)
        surf[i] = torch.as_tensor(surfaces[i : i + horizon + 1], dtype=torch.float32)
    return Paths(S=S, surf=surf, start_idx=torch.arange(n))


def run_backtest(
    paths: Paths,
    contract: Contract,
    hedger: Callable,
    instruments: InstrumentSet | None,
    cost: CostModel,
    p0: torch.Tensor,
) -> dict:
    h, q = hedger(paths, contract, instruments)
    L = terminal_loss(paths, contract, h, q, instruments, cost, p0)
    return summarize_loss(L, p0=p0, S0=paths.S[:, 0])
