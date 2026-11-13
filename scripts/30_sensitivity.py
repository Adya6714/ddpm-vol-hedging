#!/usr/bin/env python3
"""E10 sensitivity sweeps (smoke-friendly stub tables)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    args = p.parse_args()
    cfg = load_config(args.config or "configs/experiments/e10_sensitivity.yaml")
    out = ensure_dir(cfg.get("out_dir", "tables"))
    cost = pd.DataFrame({"cost_bps": cfg.get("cost_bps", [0, 2]), "note": "run full sweep after Gate 3"})
    barrier = pd.DataFrame({"barrier_mult": cfg.get("barrier_mult", [0.9]), "note": "placeholder"})
    maturity = pd.DataFrame({"maturity_days": cfg.get("maturity_days", [30]), "note": "placeholder"})
    cost.to_csv(out / "sensitivity_cost.csv", index=False)
    barrier.to_csv(out / "sensitivity_barrier.csv", index=False)
    maturity.to_csv(out / "sensitivity_maturity.csv", index=False)
    write_manifest(out / "_e10", config=cfg, seed=args.seed)
    print("wrote sensitivity_*.csv")


if __name__ == "__main__":
    main()
