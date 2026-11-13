"""W0: GBM with constant vol = inception ATM IV."""

from __future__ import annotations

import math

import numpy as np
import torch

from niftyhedge.data.splits import FoldData
from niftyhedge.gen.base import Paths, StartStates
from niftyhedge.utils.seed import set_seed


class GBMWorld:
    name = "gbm"

    def __init__(self, r: float = 0.065, dt: float = 1.0 / 252.0):
        self.r = r
        self.dt = dt

    def fit(self, fold: FoldData, seed: int) -> None:
        self._fitted = True
        self._seed = seed

    def sample_paths(
        self,
        start_states: StartStates,
        n_paths: int,
        horizon: int,
        seed: int,
    ) -> Paths:
        set_seed(seed)
        device = start_states.spot.device
        n_start = start_states.spot.shape[0]
        # sample start indices with replacement
        pick = torch.randint(0, n_start, (n_paths,), device=device)
        S0 = start_states.spot[pick]
        surf0 = start_states.surf[pick]
        idx = start_states.idx[pick]
        atm = surf0[:, 4, 2].clamp(min=1e-4)

        S = torch.zeros(n_paths, horizon + 1, device=device)
        S[:, 0] = S0
        z = torch.randn(n_paths, horizon, device=device)
        for t in range(horizon):
            S[:, t + 1] = S[:, t] * torch.exp(
                (self.r - 0.5 * atm**2) * self.dt + atm * math.sqrt(self.dt) * z[:, t]
            )
        # flat surface over path at inception level (constant vol world)
        surf = surf0[:, None, :, :].expand(-1, horizon + 1, -1, -1).clone()
        paths = Paths(S=S, surf=surf, start_idx=idx)
        paths.validate()
        return paths
