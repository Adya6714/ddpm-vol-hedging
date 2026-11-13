#!/usr/bin/env python3
"""E8 real-market walk-forward backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common, load_contract
from niftyhedge.data.surface import make_sample_surfaces
from niftyhedge.eval.backtest import overlapping_windows, run_backtest
from niftyhedge.hedge.baselines import BASELINES
from niftyhedge.hedge.instruments import CostModel
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    args = p.parse_args()
    cfg = load_config(args.config or "configs/experiments/e8_real_backtest.yaml")
    out = ensure_dir(cfg.get("out_dir", "tables"))

    surf_path = Path("data/surfaces/surfaces_v2.npy")
    if surf_path.exists() and not args.smoke:
        surfaces = np.load(surf_path)
        spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.005 * np.random.randn(len(surfaces) - 1)])
    else:
        grid = make_sample_surfaces(80 if args.smoke else 200, seed=0)
        surfaces, spot = grid.surfaces, 20000 * np.cumprod(np.r_[1.0, 1 + 0.005 * np.random.randn(len(grid.surfaces) - 1)])

    horizon = 10 if args.smoke else 30
    paths = overlapping_windows(spot, surfaces, horizon=horizon)
    cost = CostModel()
    rows = []
    for cpath in cfg.get("contracts", ["configs/contracts/doc_090.yaml"]):
        contract = load_contract(cpath)
        p0 = torch.zeros(paths.S.shape[0])
        for name, fn in BASELINES.items():
            if name not in ("no_hedge", "bs_barrier_delta_bgk", "bs_barrier_delta"):
                continue
            metrics = run_backtest(paths, contract, fn, None, cost, p0)
            rows.append({"contract": Path(cpath).stem, "hedger": name, **metrics, "n_starts": paths.S.shape[0]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "real_backtest_doc_090.csv", index=False)
    write_manifest(out / "_e8", config=cfg, seed=args.seed)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
