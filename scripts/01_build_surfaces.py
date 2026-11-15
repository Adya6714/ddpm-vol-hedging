#!/usr/bin/env python3
"""Build SVI 9x6 surfaces from Gate 1a raw options (or sample surfaces in --smoke)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common
from niftyhedge.arb.checks import report_arbitrage
from niftyhedge.data.build_surfaces import build_surfaces_from_raw
from niftyhedge.data.surface import make_sample_surfaces
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    args = p.parse_args()
    cfg = load_config(args.config or "configs/data.yaml")
    out = ensure_dir(cfg.get("surfaces_dir", "data/surfaces"))
    raw = Path(cfg.get("raw_dir", "data/raw"))

    if args.smoke:
        grid = make_sample_surfaces(120, seed=args.seed)
        surfaces, mask, dates = grid.surfaces, grid.mask, grid.dates
        forwards = pd.DataFrame({"date": [str(d) for d in dates], "forward": np.nan})
    elif (raw / "nifty_options_raw.parquet").exists() or (raw / "bhav_daily").exists():
        grid, forwards = build_surfaces_from_raw(
            raw,
            r=float(cfg.get("r", 0.065)),
            min_contracts=float(cfg.get("min_volume", 50)),
            min_price=float(cfg.get("min_price", 1.0)),
            min_dte=int(cfg.get("min_dte", 3)),
            svi_rmse_max=float(cfg.get("svi_rmse_max", 0.05)),
        )
        surfaces, mask, dates = grid.surfaces, grid.mask, grid.dates
    elif (raw / "surfaces.npy").exists():
        print("WARNING: using legacy surfaces.npy (no SVI rebuild from parquet)")
        surfaces = np.load(raw / "surfaces.npy")
        dates = np.load(raw / "surface_dates.npy", allow_pickle=True)
        mask = np.ones_like(surfaces, dtype=bool)
        forwards = pd.DataFrame({"date": [str(d) for d in dates], "forward": np.nan})
    else:
        raise FileNotFoundError(
            "No surfaces source. Upload data/raw/nifty_options_raw.parquet "
            "(from Gate 1a) and symlink to data/raw, then re-run."
        )

    np.save(out / "surfaces_v2.npy", surfaces)
    np.save(out / "surface_mask_v2.npy", mask)
    np.save(out / "surface_dates_v2.npy", dates, allow_pickle=True)
    forwards.to_parquet(out / "forwards_v2.parquet", index=False)

    arb = report_arbitrage(surfaces)
    card = pd.DataFrame(
        [
            {
                "n_days": len(dates),
                "first": str(dates[0]),
                "last": str(dates[-1]),
                "nan_frac": float(np.isnan(surfaces).mean()),
                "extrapolated_frac": float((~mask).mean()) if mask.dtype == bool else np.nan,
                "calendar_viol_frac": arb.calendar_violation_frac,
                "butterfly_viol_frac": arb.butterfly_violation_frac,
                "atm_mean": float(np.nanmean(surfaces[:, 4, 2])),
            }
        ]
    )
    ensure_dir("tables")
    card.to_csv("tables/dataset_card.csv", index=False)
    write_manifest(out / "_run", config=cfg, seed=args.seed, extra=card.iloc[0].to_dict())
    print(card.to_string(index=False))


if __name__ == "__main__":
    main()
