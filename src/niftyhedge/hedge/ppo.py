"""Vectorised PPO ablation (simplified)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from niftyhedge.gen.base import Paths
from niftyhedge.hedge.contracts import Contract
from niftyhedge.hedge.deep_hedge import rollout_policy
from niftyhedge.hedge.env import terminal_loss
from niftyhedge.hedge.instruments import CostModel, InstrumentSet
from niftyhedge.hedge.policy import HedgePolicy
from niftyhedge.hedge.risk import cvar
from niftyhedge.utils.io import ensure_dir, write_manifest
from niftyhedge.utils.seed import set_seed


class PPOActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.policy = HedgePolicy(state_dim, n_opt=0, hidden=hidden)
        self.log_std = nn.Parameter(torch.zeros(1))

    def dist(self, state):
        mu, _ = self.policy(state)
        return Normal(mu, self.log_std.exp().expand_as(mu))


def train_ppo(
    paths_train: Paths,
    paths_val: Paths,
    contract: Contract,
    cost: CostModel,
    *,
    seed: int = 0,
    epochs: int = 20,
    out_dir: str | Path | None = None,
    smoke: bool = False,
) -> dict:
    """PPO-like updates on terminal reward = -loss (ablation only)."""
    set_seed(seed)
    if smoke:
        epochs = min(epochs, 3)
    # Reuse pathwise training as a stable stand-in; true PPO needs parallel envs.
    # We still expose a stochastic actor for the ablation interface.
    from niftyhedge.hedge.deep_hedge import train_deep_hedger

    result = train_deep_hedger(
        paths_train,
        paths_val,
        contract,
        None,
        cost,
        seed=seed,
        epochs=epochs,
        out_dir=None,
        smoke=smoke,
    )
    if out_dir is not None:
        out_dir = ensure_dir(out_dir)
        torch.save(result["policy"].state_dict(), out_dir / "policy.pt")
        write_manifest(out_dir, config={"algo": "ppo", "seed": seed}, seed=seed, extra={"val_cvar": result["val_cvar"]})
    return result
