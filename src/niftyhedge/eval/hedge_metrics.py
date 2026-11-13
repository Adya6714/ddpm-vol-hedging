"""Hedging evaluation metrics."""

from __future__ import annotations

import torch

from niftyhedge.hedge.risk import cvar, var


def summarize_loss(loss: torch.Tensor, p0: torch.Tensor | None = None, S0: torch.Tensor | None = None) -> dict:
    loss = loss.detach()
    out = {
        "mean": float(loss.mean()),
        "std": float(loss.std()),
        "var95": float(var(loss, 0.95)),
        "cvar95": float(cvar(loss, 0.95)),
        "cvar99": float(cvar(loss, 0.99)),
    }
    if S0 is not None:
        out["cvar95_pct_S0"] = float(cvar(loss / S0, 0.95) * 100)
    if p0 is not None:
        out["cvar95_pct_p0"] = float(cvar(loss / p0.clamp(min=1e-6), 0.95) * 100)
    return out
