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

    if not args.smoke and not (surf_dir / "surfaces_v2.npy").exists():
        raise FileNotFoundError(
            f"{surf_dir / 'surfaces_v2.npy'} missing. Run the Restore cell (or scripts/01_build_surfaces.py) first; "
            "refusing to fall back to synthetic surfaces for a real run."
        )
    if args.smoke:
        grid = make_sample_surfaces(120, seed=args.seed)
        surfaces, dates = grid.surfaces, grid.dates
        spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.005 * np.random.randn(len(dates) - 1)])
        vix = np.clip(14 + np.random.randn(len(dates)), 8, 40)
        test_years = [2020]
    else:
        surfaces = np.load(surf_dir / "surfaces_v2.npy")
        dates = np.load(surf_dir / "surface_dates_v2.npy", allow_pickle=True)
        spot_path = surf_dir / "spot_v2.parquet"
        if not spot_path.exists():
            raise FileNotFoundError(
                f"{spot_path} missing - re-run scripts/01_build_surfaces.py (it now writes the "
                "futures-implied spot series). Refusing to fall back to a random walk."
            )
        import pandas as pd

        sp = pd.read_parquet(spot_path)
        if list(sp["date"].astype(str)) != [str(d) for d in dates]:
            raise ValueError("spot_v2.parquet dates do not match surface dates")
        spot = sp["spot"].to_numpy(dtype=float)
        # India VIX is not in bhavcopy; it is itself a ~30-day NIFTY option-implied
        # vol, so the 30-day ATM IV of our own surface is used as the VIX proxy
        # (build_conditioning lags it, so no same-day leakage).
        vix = 100.0 * surfaces[:, 4, 2]
        test_years = cfg.get("test_years")

    C = build_conditioning(surfaces, dates, spot, vix, lag=int(cfg.get("lag", 1)))
    folds = make_walk_forward_folds(
        surfaces, C, dates, spot, folds_dir, test_years=test_years, val_months=int(cfg.get("val_months", 6))
    )
    write_manifest(folds_dir, config=cfg, seed=args.seed, extra={"n_folds": len(folds)})
    print(f"Wrote {len(folds)} folds to {folds_dir}")


if __name__ == "__main__":
    main()
