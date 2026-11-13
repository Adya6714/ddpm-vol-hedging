#!/usr/bin/env python3
"""Build SVI surfaces (or sample surfaces in --smoke)."""

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
from niftyhedge.data.surface import make_sample_surfaces, MONEYNESS, TENORS_DAYS
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    args = p.parse_args()
    cfg = load_config(args.config or "configs/data.yaml")
    out = ensure_dir(cfg.get("surfaces_dir", "data/surfaces"))

    if args.smoke:
        grid = make_sample_surfaces(120, seed=args.seed)
        surfaces, mask, dates = grid.surfaces, grid.mask, grid.dates
    else:
        # If rebuilt parquet exists with full pipeline, use it; else fall back to legacy npy
        raw = Path(cfg.get("raw_dir", "data/raw"))
        legacy = raw / "surfaces.npy"
        if legacy.exists():
            surfaces = np.load(legacy)
            dates = np.load(raw / "surface_dates.npy", allow_pickle=True)
            mask = np.ones_like(surfaces, dtype=bool)
            print("WARNING: using legacy surfaces.npy; re-download + SVI fit for production")
        else:
            raise FileNotFoundError("No surfaces source; run download then full SVI pipeline")

    np.save(out / "surfaces_v2.npy", surfaces)
    np.save(out / "surface_mask_v2.npy", mask)
    np.save(out / "surface_dates_v2.npy", dates, allow_pickle=True)
    # forwards stub
    pd.DataFrame({"date": [str(d) for d in dates], "forward": np.nan}).to_parquet(out / "forwards_v2.parquet")

    arb = report_arbitrage(surfaces)
    card = pd.DataFrame(
        [
            {
                "n_days": len(dates),
                "first": str(dates[0]),
                "last": str(dates[-1]),
                "nan_frac": float(np.isnan(surfaces).mean()),
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
