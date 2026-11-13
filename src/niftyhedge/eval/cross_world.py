"""Train-world × test-world evaluation matrix."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import torch

from niftyhedge.gen.base import Paths
from niftyhedge.hedge.contracts import Contract
from niftyhedge.hedge.env import terminal_loss
from niftyhedge.hedge.instruments import CostModel, InstrumentSet
from niftyhedge.hedge.risk import cvar


def evaluate_matrix(
    hedgers: dict[str, Callable],
    test_paths: dict[str, Paths],
    contract: Contract,
    instruments: InstrumentSet | None,
    cost: CostModel,
) -> pd.DataFrame:
    rows = []
    for hname, hedger in hedgers.items():
        for wname, paths in test_paths.items():
            h, q = hedger(paths, contract, instruments)
            p0 = torch.zeros(paths.S.shape[0], device=paths.S.device)
            L = terminal_loss(paths, contract, h, q, instruments, cost, p0)
            rows.append({"hedger": hname, "test_world": wname, "cvar95": float(cvar(L))})
    return pd.DataFrame(rows).pivot(index="hedger", columns="test_world", values="cvar95")
