"""W2: PCA-VAR on log-IV + return."""

from __future__ import annotations

import numpy as np
import torch
from numpy.linalg import pinv

from niftyhedge.data.splits import FoldData
from niftyhedge.gen.base import Paths, StartStates
from niftyhedge.utils.seed import set_seed


class PCAVARWorld:
    name = "pcavar"

    def __init__(self, n_factors: int = 3, r: float = 0.065, dt: float = 1 / 252):
        self.n_factors = n_factors
        self.r = r
        self.dt = dt
        self.mean_flat: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.residuals: np.ndarray | None = None

    def fit(self, fold: FoldData, seed: int) -> None:
        X = np.log(np.clip(fold.X_train, 1e-6, None)).reshape(len(fold.X_train), -1)
        spot = fold.spot_train
        ret = np.zeros(len(spot))
        ret[1:] = np.log(spot[1:] / spot[:-1])
        self.mean_flat = X.mean(axis=0)
        Xc = X - self.mean_flat
        # PCA via SVD
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components = Vt[: self.n_factors]
        scores = Xc @ self.components.T  # [T, F]
        # state = [scores, ret]
        Z = np.column_stack([scores, ret])
        Z0, Z1 = Z[:-1], Z[1:]
        self.A = Z1.T @ Z0 @ pinv(Z0.T @ Z0)
        pred = (self.A @ Z0.T).T
        self.residuals = Z1 - pred
        self._fitted = True

    def sample_paths(self, start_states: StartStates, n_paths: int, horizon: int, seed: int) -> Paths:
        set_seed(seed)
        assert self.components is not None and self.A is not None and self.residuals is not None
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

        rng = np.random.default_rng(seed)
        comps = self.components
        mean = self.mean_flat
        A = self.A
        resid = self.residuals

        for i in range(n_paths):
            flat = np.log(np.clip(surf0[i].cpu().numpy().reshape(-1), 1e-6, None))
            scores = (flat - mean) @ comps.T
            z = np.concatenate([scores, [0.0]])
            for t in range(horizon):
                eps = resid[rng.integers(0, len(resid))]
                z = A @ z + eps
                scores, ret = z[:-1], z[-1]
                flat = mean + scores @ comps
                iv = np.exp(flat).reshape(9, 6)
                iv = np.clip(iv, 0.05, 1.5)
                surf[i, t + 1] = torch.as_tensor(iv, dtype=surf.dtype, device=device)
                atm = float(iv[4, 2])
                # blend VAR return with GBM using ATM for stability
                S[i, t + 1] = S[i, t] * math_exp_ret(ret, atm, self.r, self.dt, rng)

        paths = Paths(S=S, surf=surf, start_idx=idx)
        paths.validate()
        return paths


def math_exp_ret(ret, atm, r, dt, rng) -> float:
    # use VAR residual return directly (already a daily log return)
    return float(np.exp(ret))
