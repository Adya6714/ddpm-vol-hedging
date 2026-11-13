"""W4: Conditional DDPM for (log-return, delta log-IV surface)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from niftyhedge.data.splits import FoldData
from niftyhedge.gen.base import Paths, StartStates
from niftyhedge.utils.seed import set_seed


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


class Denoiser(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, t_frac, cond):
        inp = torch.cat([x, t_frac, cond], dim=-1)
        return self.net(inp)


def cosine_alpha_bars(T: int, device) -> torch.Tensor:
    s = 0.008
    steps = torch.arange(T + 1, device=device, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    return f / f[0]


class DDPMWorld:
    name = "ddpm"

    def __init__(
        self,
        timesteps: int = 100,
        hidden: int = 128,
        epochs: int = 50,
        lr: float = 1e-3,
        r: float = 0.065,
        device: str | None = None,
    ):
        self.T = timesteps
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.r = r
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Denoiser | None = None
        self.ema: EMA | None = None
        self.cond_dim = 0
        self.x_dim = 54 + 1  # dlogIV flat + return

    def fit(self, fold: FoldData, seed: int) -> None:
        set_seed(seed)
        X = fold.X_train
        C = fold.C_train
        spot = fold.spot_train
        # targets: day-over-day change
        dlog = np.log(np.clip(X[1:], 1e-6, None)) - np.log(np.clip(X[:-1], 1e-6, None))
        ret = np.log(spot[1:] / spot[:-1]).reshape(-1, 1)
        Y = np.concatenate([dlog.reshape(len(dlog), -1), ret], axis=1).astype(np.float32)
        # condition with previous day's C (already lag-1 in features); align to Y rows
        Cond = C[1:].astype(np.float32)
        n = min(len(Y), len(Cond))
        Y, Cond = Y[:n], Cond[:n]

        self.cond_dim = Cond.shape[1]
        self.x_dim = Y.shape[1]
        self.model = Denoiser(self.x_dim, self.cond_dim, self.hidden).to(self.device)
        self.ema = EMA(self.model)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ab = cosine_alpha_bars(self.T, self.device)

        Yt = torch.as_tensor(Y, device=self.device)
        Ct = torch.as_tensor(Cond, device=self.device)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, 64):
                idx = perm[i : i + 64]
                x0 = Yt[idx]
                c = Ct[idx]
                t = torch.randint(0, self.T, (len(idx),), device=self.device)
                a_bar = ab[t + 1].unsqueeze(-1)
                eps = torch.randn_like(x0)
                xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
                t_frac = (t.float() / self.T).unsqueeze(-1)
                pred = self.model(xt, t_frac, c)
                loss = ((pred - eps) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.ema.update(self.model)
        self.ema.copy_to(self.model)
        self.model.eval()
        self._ab = ab
        self._fitted = True

    @torch.no_grad()
    def _sample_step(self, cond: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        n = cond.shape[0]
        x = torch.randn(n, self.x_dim, device=self.device)
        ab = self._ab
        for t in reversed(range(self.T)):
            t_frac = torch.full((n, 1), t / self.T, device=self.device)
            eps = self.model(x, t_frac, cond)
            a_bar = ab[t + 1]
            a_bar_prev = ab[t]
            x0 = (x - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar).clamp(min=1e-6)
            # no clamp on x0 (fixes G7-style bias from notebook)
            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1 - a_bar_prev) * noise
            else:
                x = x0
        return x

    def sample_paths(self, start_states: StartStates, n_paths: int, horizon: int, seed: int) -> Paths:
        set_seed(seed)
        assert self.model is not None
        device = self.device
        n_start = start_states.spot.shape[0]
        pick = torch.randint(0, n_start, (n_paths,), device=start_states.spot.device)
        S0 = start_states.spot[pick].to(device)
        surf0 = start_states.surf[pick].to(device)
        idx = start_states.idx[pick].to(device)

        S = torch.zeros(n_paths, horizon + 1, device=device)
        surf = torch.zeros(n_paths, horizon + 1, 9, 6, device=device)
        S[:, 0] = S0
        surf[:, 0] = surf0

        # simple cond: flatten current surface + zeros for other features (smoke-safe)
        for t in range(horizon):
            flat = surf[:, t].reshape(n_paths, -1)
            # pad/truncate to cond_dim
            cond = torch.zeros(n_paths, self.cond_dim, device=device)
            n_copy = min(flat.shape[1], self.cond_dim)
            cond[:, :n_copy] = flat[:, :n_copy]
            y = self._sample_step(cond)
            dlog = y[:, :54].reshape(n_paths, 9, 6).clamp(-0.5, 0.5)
            ret = y[:, 54].clamp(-0.1, 0.1)
            next_surf = (surf[:, t] * torch.exp(dlog)).clamp(0.05, 1.5)
            next_S = S[:, t] * torch.exp(ret)
            # hard NaN guard (untrained smoke models can still explode)
            bad = ~torch.isfinite(next_S) | ~torch.isfinite(next_surf).all(dim=(-1, -2))
            if bad.any():
                next_S = torch.where(bad, S[:, t], next_S)
                next_surf = torch.where(bad[:, None, None], surf[:, t], next_surf)
            surf[:, t + 1] = next_surf
            S[:, t + 1] = next_S

        paths = Paths(S=S.cpu(), surf=surf.cpu(), start_idx=idx.cpu())
        paths.validate()
        return paths

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model else None,
                "cond_dim": self.cond_dim,
                "x_dim": self.x_dim,
                "T": self.T,
                "hidden": self.hidden,
            },
            path,
        )
