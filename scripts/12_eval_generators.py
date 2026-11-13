#!/usr/bin/env python3
"""Generator quality tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common
from niftyhedge.eval.gen_metrics import atm_stats, marginal_distances
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--ablations", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config or "configs/experiments/e5_gen_quality.yaml")
    out = ensure_dir(cfg.get("out_dir", "tables"))
    rows = []
    world_rows = []
    for w in cfg.get("worlds", ["gbm"]):
        # find any fold paths
        matches = list(Path("data/paths").glob(f"{w}/fold_*/paths_test.pt"))
        if not matches:
            if args.smoke:
                continue
            print(f"skip {w}: no paths")
            continue
        blob = torch.load(matches[0], map_location="cpu", weights_only=False)
        surf = blob["surf"].numpy()
        atm = surf[:, -1, 4, 2]
        st = atm_stats(surf[:, -1])
        world_rows.append({"world": w, **st})
        # persistence baseline vs end-of-path ATM
        persist = surf[:, 0, 4, 2]
        rows.append({"world": w, "metric": "atm_w1_vs_start", **marginal_distances(persist, atm)})
    pd.DataFrame(rows).to_csv(out / "gen_quality.csv", index=False)
    pd.DataFrame(world_rows).to_csv(out / "world_stats.csv", index=False)
    write_manifest(out / "_gen_eval", config=cfg, seed=args.seed)
    print(pd.DataFrame(world_rows).to_string(index=False))


if __name__ == "__main__":
    main()
