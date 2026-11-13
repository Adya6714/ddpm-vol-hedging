"""Shared CLI helpers for scripts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from niftyhedge.data.splits import FoldData, load_fold, make_walk_forward_folds
from niftyhedge.data.surface import make_sample_surfaces
from niftyhedge.data.features import build_conditioning
from niftyhedge.gen.gbm import GBMWorld
from niftyhedge.gen.heston import HestonWorld
from niftyhedge.gen.pcavar import PCAVARWorld
from niftyhedge.gen.bootstrap import BootstrapWorld
from niftyhedge.gen.ddpm import DDPMWorld
from niftyhedge.hedge.contracts import Contract
from niftyhedge.utils.config import load_config
import numpy as np


WORLDS = {
    "gbm": GBMWorld,
    "heston": HestonWorld,
    "pcavar": PCAVARWorld,
    "bootstrap": BootstrapWorld,
    "ddpm": DDPMWorld,
}


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")


def parse_seeds(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x != ""]


def parse_folds(s: str, n: int = 6) -> list[int]:
    if s == "all":
        return list(range(n))
    return [int(x) for x in str(s).split(",") if x != ""]


def load_contract(path: str) -> Contract:
    cfg = load_config(path)
    return Contract(**cfg)


def ensure_smoke_folds(folds_dir: Path) -> None:
    folds_dir = Path(folds_dir)
    if (folds_dir / "fold_0").exists():
        return
    grid = make_sample_surfaces(120, seed=0)
    spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.01 * np.random.randn(119)])
    vix = 14 + 2 * np.random.randn(120)
    C = build_conditioning(grid.surfaces, grid.dates, spot, vix, lag=1)
    make_walk_forward_folds(grid.surfaces, C, grid.dates, spot, folds_dir, test_years=[2020])


def get_world(name: str, cfg: dict | None = None, smoke: bool = False):
    cfg = cfg or {}
    if name == "ddpm" and smoke:
        return DDPMWorld(timesteps=10, hidden=64, epochs=2)
    if name == "ddpm":
        return DDPMWorld(
            timesteps=int(cfg.get("timesteps", 100)),
            hidden=int(cfg.get("hidden", 128)),
            epochs=int(cfg.get("epochs", 50)),
            lr=float(cfg.get("lr", 1e-3)),
        )
    if name == "pcavar":
        return PCAVARWorld(n_factors=int(cfg.get("n_factors", 3)))
    if name == "bootstrap":
        return BootstrapWorld(block=int(cfg.get("block", 5)))
    return WORLDS[name]()


def save_world(world, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(world, f)


def load_world(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)
