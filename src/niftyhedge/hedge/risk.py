"""Risk measures and Rockafellar–Uryasev CVaR objective."""

from __future__ import annotations

import torch


def cvar(loss: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Empirical CVaR of terminal loss (positive = bad)."""
    if loss.ndim != 1:
        raise ValueError("loss must be 1-D")
    n = loss.numel()
    k = max(int(np_ceil((1.0 - alpha) * n)), 1)
    # mean of the worst k losses
    worst, _ = torch.topk(loss, k)
    return worst.mean()


def np_ceil(x: float) -> int:
    import math

    return int(math.ceil(x))


def var(loss: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    return torch.quantile(loss, alpha)


def entropic(loss: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return (1.0 / lam) * torch.log(torch.mean(torch.exp(lam * loss)))


def ru_objective(loss: torch.Tensor, w: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Differentiable Rockafellar–Uryasev: w + E[(L-w)+]/(1-alpha)."""
    return w + torch.relu(loss - w).mean() / (1.0 - alpha)
