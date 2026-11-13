"""Barrier option contracts and discrete monitoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Contract:
    kind: str  # "DOC" down-and-out call, "UOC" up-and-out call, "VANILLA_CALL"
    strike_mult: float = 1.0  # K = strike_mult * S0
    barrier_mult: float = 0.90  # B = barrier_mult * S0
    maturity_days: int = 30


def barrier_payoff(paths_S: torch.Tensor, contract: Contract) -> torch.Tensor:
    """Terminal payoff under discrete daily monitoring. paths_S: [n, H+1]."""
    S0 = paths_S[:, 0]
    K = contract.strike_mult * S0
    B = contract.barrier_mult * S0
    ST = paths_S[:, -1]
    if contract.kind == "VANILLA_CALL":
        return torch.clamp(ST - K, min=0.0)
    if contract.kind == "DOC":
        knocked = (paths_S <= B[:, None]).any(dim=1)
        payoff = torch.clamp(ST - K, min=0.0)
        return torch.where(knocked, torch.zeros_like(payoff), payoff)
    if contract.kind == "UOC":
        knocked = (paths_S >= B[:, None]).any(dim=1)
        payoff = torch.clamp(ST - K, min=0.0)
        return torch.where(knocked, torch.zeros_like(payoff), payoff)
    raise ValueError(f"Unknown contract kind: {contract.kind}")


def knock_mask(paths_S: torch.Tensor, contract: Contract) -> torch.Tensor:
    """Boolean [n, H+1]: True once knocked (inclusive of knock day)."""
    S0 = paths_S[:, 0]
    B = contract.barrier_mult * S0
    n, Hp1 = paths_S.shape
    alive = torch.ones(n, Hp1, dtype=torch.bool, device=paths_S.device)
    if contract.kind == "VANILLA_CALL":
        return ~alive  # never knocked
    knocked = torch.zeros(n, dtype=torch.bool, device=paths_S.device)
    out = torch.zeros(n, Hp1, dtype=torch.bool, device=paths_S.device)
    for t in range(Hp1):
        if contract.kind == "DOC":
            knocked = knocked | (paths_S[:, t] <= B)
        else:
            knocked = knocked | (paths_S[:, t] >= B)
        out[:, t] = knocked
    return out
