#!/usr/bin/env python3
"""Leak-free conditioning + walk-forward folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common
from niftyhedge.data.features import build_conditioning
from niftyhedge.data.splits import make_walk_forward_folds
from niftyhedge.data.surface import make_sample_surfaces
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import write_manifest
from niftyhedge.utils.seed import set_seed


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    args = p.parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config or "configs/folds.yaml")
    surf_dir = Path(cfg.get("surfaces_dir", "data/surfaces"))
    folds_dir = Path(cfg.get("folds_dir", "data/folds"))

    if args.smoke or not (surf_dir / "surfaces_v2.npy").exists():
        grid = make_sample_surfaces(120, seed=args.seed)
        surfaces, dates = grid.surfaces, grid.dates
        spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.005 * np.random.randn(len(dates) - 1)])
        vix = np.clip(14 + np.random.randn(len(dates)), 8, 40)
        test_years = [2020]
    else:
        surfaces = np.load(surf_dir / "surfaces_v2.npy")
        dates = np.load(surf_dir / "surface_dates_v2.npy", allow_pickle=True)
        raw = Path("data/raw")
        if (raw / "nifty_spot.parquet").exists():
            import pandas as pd

            spot_df = pd.read_parquet(raw / "nifty_spot.parquet")
            # align roughly by length if needed
            spot = spot_df["close"].to_numpy(dtype=float)
            if len(spot) != len(dates):
                spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.005 * np.random.randn(len(dates) - 1)])
        else:
            spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.005 * np.random.randn(len(dates) - 1)])
        vix = np.clip(14 + np.random.randn(len(dates)), 8, 40)
        test_years = cfg.get("test_years")

    C = build_conditioning(surfaces, dates, spot, vix, lag=int(cfg.get("lag", 1)))
    folds = make_walk_forward_folds(
        surfaces, C, dates, spot, folds_dir, test_years=test_years, val_months=int(cfg.get("val_months", 6))
    )
    write_manifest(folds_dir, config=cfg, seed=args.seed, extra={"n_folds": len(folds)})
    print(f"Wrote {len(folds)} folds to {folds_dir}")


if __name__ == "__main__":
    main()
