#!/usr/bin/env python3
"""E7 cross-world matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common, load_contract
from niftyhedge.eval.cross_world import evaluate_matrix
from niftyhedge.eval.plots import save_heatmap
from niftyhedge.gen.base import Paths
from niftyhedge.hedge.baselines import BASELINES
from niftyhedge.hedge.instruments import CostModel
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    args = p.parse_args()
    cfg = load_config(args.config or "configs/experiments/e7_cross_world.yaml")
    contract = load_contract(cfg.get("contract", "configs/contracts/doc_090.yaml"))
    cost = CostModel()
    test_paths = {}
    for w in cfg.get("worlds", ["gbm"]):
        matches = list(Path("data/paths").glob(f"{w}/fold_*/paths_test.pt"))
        if not matches:
            continue
        blob = torch.load(matches[0], map_location="cpu", weights_only=False)
        n = 32 if args.smoke else min(10000, blob["S"].shape[0])
        test_paths[w] = Paths(S=blob["S"][:n], surf=blob["surf"][:n], start_idx=blob["start_idx"][:n])
    if not test_paths:
        print("no test paths; skip")
        ensure_dir("tables").joinpath("cross_world_matrix_I1.csv").write_text("hedger\n")
        return
    hedgers = {name: fn for name, fn in BASELINES.items() if name in ("no_hedge", "bs_barrier_delta_bgk")}
    df = evaluate_matrix(hedgers, test_paths, contract, None, cost)
    out = ensure_dir(cfg.get("out_dir", "tables"))
    df.to_csv(out / "cross_world_matrix_I1.csv")
    save_heatmap(df, "figures/cross_world_I1.pdf", title="CVaR95 cross-world")
    write_manifest(out / "_e7", config=cfg, seed=args.seed)
    print(df)


if __name__ == "__main__":
    main()
