"""Thin wrapper used by scripts to fit/save a DDPM world."""

from __future__ import annotations

from pathlib import Path

from niftyhedge.data.splits import FoldData
from niftyhedge.gen.ddpm import DDPMWorld


def train_ddpm(fold: FoldData, out_dir: str | Path, seed: int, smoke: bool = False) -> DDPMWorld:
    kwargs = dict(timesteps=20, hidden=64, epochs=3) if smoke else dict()
    world = DDPMWorld(**kwargs)
    world.fit(fold, seed)
    world.save(Path(out_dir) / "ddpm.pt")
    return world
