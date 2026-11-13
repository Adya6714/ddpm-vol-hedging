"""Common World interface and Paths dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from niftyhedge.data.splits import FoldData
from niftyhedge.utils.io import assert_finite


@dataclass
class StartStates:
    """Real historical start states for path generation."""

    spot: torch.Tensor  # [n]
    surf: torch.Tensor  # [n, 9, 6]
    idx: torch.Tensor  # [n] index into fold dates


@dataclass
class Paths:
    S: torch.Tensor  # [n, H+1]
    surf: torch.Tensor  # [n, H+1, 9, 6]
    start_idx: torch.Tensor  # [n]

    def validate(self) -> None:
        assert_finite(self.S, "Paths.S")
        assert_finite(self.surf, "Paths.surf")
        if (self.surf <= 0).any():
            raise ValueError("Paths.surf must be strictly positive")
        if self.S.shape[0] != self.surf.shape[0] or self.S.shape[1] != self.surf.shape[1]:
            raise ValueError("S and surf time dimensions must match")


class World(Protocol):
    name: str

    def fit(self, fold: FoldData, seed: int) -> None: ...

    def sample_paths(
        self,
        start_states: StartStates,
        n_paths: int,
        horizon: int,
        seed: int,
    ) -> Paths: ...


def start_states_from_fold(fold: FoldData, split: str = "train", device="cpu") -> StartStates:
    X = getattr(fold, f"X_{split}")
    spot = getattr(fold, f"spot_{split}")
    n = len(spot)
    return StartStates(
        spot=torch.as_tensor(spot, dtype=torch.float32, device=device),
        surf=torch.as_tensor(X, dtype=torch.float32, device=device),
        idx=torch.arange(n, device=device),
    )
