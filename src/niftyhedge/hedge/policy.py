"""MLP hedging policy."""

from __future__ import annotations

import torch
import torch.nn as nn


class HedgePolicy(nn.Module):
    """Maps state features -> underlying (and optional option) positions."""

    def __init__(self, state_dim: int, n_opt: int = 0, hidden: int = 128, residual: bool = True,
                 max_correction: float = 1.0):
        """residual=True: h = BS-barrier delta (last state feature) + learned correction.

        The output layer is zero-initialised, so an untrained policy *is* the
        BGK barrier delta hedge with no options; training can only move away
        from it if that lowers CVaR. This is the hybrid model-plus-network design.
        """
        super().__init__()
        self.n_opt = n_opt
        self.residual = residual
        self.max_correction = max_correction
        out = 1 + n_opt
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out),
        )
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = self.net(x)
        if self.residual:
            h = x[..., -1] + self.max_correction * torch.tanh(y[..., 0])
        else:
            h = 2.0 * torch.tanh(y[..., 0])
        if self.n_opt > 0:
            q = torch.tanh(y[..., 1:])
            return h, q
        return h, None


def build_state(
    S: torch.Tensor,
    surf: torch.Tensor,
    t: int,
    maturity: int,
    K: torch.Tensor,
    B: torch.Tensor,
    h_prev: torch.Tensor,
    include_bs_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal state at time t. Does not read surf[:, t+1:]."""
    log_sk = torch.log(S[:, t] / K)
    log_sb = torch.log(S[:, t] / B)
    tau = torch.full_like(S[:, t], (maturity - t) / maturity)
    atm = surf[:, t, 4, 2]
    skew = surf[:, t, 3, 2] - surf[:, t, 5, 2]
    term = surf[:, t, 4, 4] - surf[:, t, 4, 2]
    feats = [log_sk, log_sb, tau, h_prev, atm, skew, term]
    if include_bs_delta is not None:
        feats.append(include_bs_delta)
    return torch.stack(feats, dim=-1)
