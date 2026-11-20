#!/usr/bin/env python3
"""E7 - cross-world matrix: hedger trained in world A, tested in world B (and on real data).

Rows: classical baselines + deep[W] (mean CVaR over seeds) for each training world W.
Columns: each simulated world's held-out test paths for the same fold, plus `real`
(real NIFTY windows in that fold's test year). Diagonal = in-world performance;
the `real` column is what the paper's claims rest on.
Units: CVaR95 of loss / S0 in basis points of notional.
"""

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
from niftyhedge.data.splits import load_fold
from niftyhedge.eval.plots import save_heatmap
from niftyhedge.eval.real import discover_runs, evaluate_all, load_real_market, real_windows
from niftyhedge.eval.stats import cvar_np
from niftyhedge.gen.base import Paths
from niftyhedge.hedge.instruments import CostModel, instrument_set_for
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--fold", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config or "configs/experiments/e7_cross_world.yaml")
    contract_path = cfg.get("contract", "configs/contracts/doc_090.yaml")
    contract = load_contract(contract_path)
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 5))
    cost = CostModel(underlying_bps=float(cfg.get("cost_bps", 2.0)))
    n_test = 64 if args.smoke else int(cfg.get("n_test", 20000))

    test_sets: dict[str, Paths] = {}
    for w in cfg.get("worlds", ["gbm"]):
        f = Path(f"data/paths/{w}/fold_{fold}/paths_test.pt")
        if not f.exists():
            print(f"skip test world {w}: {f} missing")
            continue
        blob = torch.load(f, map_location="cpu", weights_only=False)
        n = min(n_test, blob["S"].shape[0])
        test_sets[w] = Paths(S=blob["S"][:n].float(), surf=blob["surf"][:n].float(), start_idx=blob["start_idx"][:n])
    try:
        dates, spot, surfaces = load_real_market()
        test_sets["real"], _ = real_windows(dates, spot, surfaces, load_fold(Path("data/folds") / f"fold_{fold}").dates_test,
                                            contract.maturity_days)
    except (FileNotFoundError, ValueError) as e:
        print(f"skip real column: {e}")

    out = ensure_dir(cfg.get("out_dir", "tables"))
    for instr in cfg.get("instruments", ["I1"]):
        instruments = instrument_set_for(instr, contract.barrier_mult)
        runs = discover_runs("models/hedge", Path(contract_path).stem, instr, fold)
        if not runs:
            print(f"{instr}: no trained hedgers for fold {fold}; classical rows only")
        rows = []
        for tw, paths in test_sets.items():
            res = evaluate_all(paths, contract, instruments, cost, runs, None)
            for hedger, seeds in res.items():
                cv = [cvar_np(v) for v in seeds.values()]
                rows.append({"hedger": hedger, "test_world": tw, "cvar95_bp": 1e4 * np.mean(cv),
                             "seed_sd_bp": 1e4 * np.std(cv), "n_seeds": len(cv), "n_paths": paths.S.shape[0]})
        long = pd.DataFrame(rows)
        long.to_csv(out / f"cross_world_{instr}_long.csv", index=False)
        mat = long.pivot(index="hedger", columns="test_world", values="cvar95_bp")
        cols = [c for c in mat.columns if c != "real"] + (["real"] if "real" in mat.columns else [])
        mat = mat[cols]
        mat.to_csv(out / f"cross_world_matrix_{instr}.csv")
        save_heatmap(mat, f"figures/cross_world_{instr}.pdf", title=f"CVaR95 (bp of notional), {instr}, fold {fold}")
        print(f"\n== {instr}  fold {fold}  CVaR95 bp of notional (rows: hedger / training world) ==")
        print(mat.round(1).to_string())
    write_manifest(out / "_e7", config=cfg, seed=args.seed)


if __name__ == "__main__":
    main()
