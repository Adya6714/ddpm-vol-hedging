"""Autoregressive path generation helper."""

from __future__ import annotations

from niftyhedge.gen.base import Paths, StartStates, World


def generate_paths(
    world: World,
    start_states: StartStates,
    n_paths: int,
    horizon: int,
    seed: int,
) -> Paths:
    return world.sample_paths(start_states, n_paths, horizon, seed)
