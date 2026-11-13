"""W3: Stationary block bootstrap of (return, surface-change) sequences."""

from __future__ import annotations

import numpy as np
import torch

from niftyhedge.data.splits import FoldData
from niftyhedge.gen.base import Paths, StartStates
from niftyhedge.utils.seed import set_seed


class BootstrapWorld:
    name = "bootstrap"

    def __init__(self, block: int = 5, r: float = 0.065):
        self.block = block
        self.r = r
        self.returns: np.ndarray | None = None
        self.dsurf: np.ndarray | None = None

    def fit(self, fold: FoldData, seed: int) -> None:
        X = fold.X_train
        spot = fold.spot_train
        ret = np.zeros(len(spot) - 1)
        ret[:] = np.log(spot[1:] / spot[:-1])
        dsurf = np.log(np.clip(X[1:], 1e-6, None)) - np.log(np.clip(X[:-1], 1e-6, None))
        self.returns = ret
        self.dsurf = dsurf
        self._fitted = True

    def sample_paths(self, start_states: StartStates, n_paths: int, horizon: int, seed: int) -> Paths:
        set_seed(seed)
        assert self.returns is not None and self.dsurf is not None
        device = start_states.spot.device
        n_start = start_states.spot.shape[0]
        pick = torch.randint(0, n_start, (n_paths,), device=device)
        S0 = start_states.spot[pick]
        surf0 = start_states.surf[pick]
        idx = start_states.idx[pick]

        S = torch.zeros(n_paths, horizon + 1, device=device)
        surf = torch.zeros(n_paths, horizon + 1, 9, 6, device=device)
        S[:, 0] = S0
        surf[:, 0] = surf0

        rng = np.random.default_rng(seed + 1)
        Tavail = len(self.returns)
        for i in range(n_paths):
            log_s = np.log(np.clip(surf0[i].cpu().numpy(), 1e-6, None))
            t = 0
            while t < horizon:
                start = int(rng.integers(0, max(Tavail - self.block, 1)))
                for b in range(self.block):
                    if t >= horizon:
                        break
                    j = min(start + b, Tavail - 1)
                    log_s = log_s + self.dsurf[j]
                    iv = np.clip(np.exp(log_s), 0.05, 1.5)
                    surf[i, t + 1] = torch.as_tensor(iv, dtype=surf.dtype, device=device)
                    S[i, t + 1] = S[i, t] * float(np.exp(self.returns[j]))
                    t += 1

        paths = Paths(S=S, surf=surf, start_idx=idx)
        paths.validate()
        return paths
