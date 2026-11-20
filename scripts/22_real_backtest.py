#!/usr/bin/env python3
"""E8 - real-NIFTY walk-forward backtest of every trained hedger (the paper's main table).

For each fold k, hedgers trained on fold k are applied to REAL 30-day windows
(futures-implied spot + market IV surfaces) starting in fold k's test year.
Folds are pooled, so the test spans 2020 - Mar 2026 with no look-ahead.

Outputs (tables/):
  real_backtest_losses.parquet  per-window losses (loss / S0) for every hedger/seed
  real_backtest_summary.csv     CVaR95/99, mean, std in bp of notional, block-bootstrap
                                CIs, paired difference vs BGK delta, seed spread
The previous version scored only classical baselines on a *random-walk* spot
with premium 0; it never evaluated a trained hedger.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common, ensure_smoke_folds, load_contract, n_folds
from niftyhedge.data.splits import load_fold
from niftyhedge.eval.real import discover_runs, evaluate_all, load_real_market, real_windows
from niftyhedge.eval.stats import cvar_np, paired_block_bootstrap
from niftyhedge.hedge.instruments import CostModel, instrument_set_for
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, write_manifest

REF = "bs_barrier_delta_bgk"


def summarise(losses: pd.DataFrame, block: int, n_boot: int) -> pd.DataFrame:
    rows = []
    for (contract, instr), g in losses.groupby(["contract", "instruments"]):
        ref = g[g.hedger == REF].sort_values(["date"])["loss"].to_numpy()
        for hedger, gh in g.groupby("hedger"):
            per_seed = {s: gs.sort_values("date")["loss"].to_numpy() for s, gs in gh.groupby("seed")}
            # seed-averaged position is not meaningful for CVaR; average the *metric* over seeds
            cv_seeds = [cvar_np(v) for v in per_seed.values()]
            # CI / paired test on the seed whose CVaR is the median (representative run)
            rep = list(per_seed.values())[int(np.argsort(cv_seeds)[len(cv_seeds) // 2])]
            bs = paired_block_bootstrap(rep, ref, stat=cvar_np, block=block, n_boot=n_boot)
            all_l = np.concatenate(list(per_seed.values()))
            rows.append({
                "contract": contract, "instruments": instr, "hedger": hedger,
                "n_windows": len(rep), "n_seeds": len(per_seed),
                "cvar95_bp": 1e4 * float(np.mean(cv_seeds)),
                "cvar95_bp_seed_sd": 1e4 * float(np.std(cv_seeds)),
                "cvar95_bp_lo": 1e4 * bs["lo"], "cvar95_bp_hi": 1e4 * bs["hi"],
                "cvar99_bp": 1e4 * cvar_np(all_l, 0.99),
                "mean_bp": 1e4 * float(all_l.mean()), "std_bp": 1e4 * float(all_l.std()),
                "diff_vs_bgk_bp": 1e4 * bs["diff"], "diff_lo_bp": 1e4 * bs["diff_lo"],
                "diff_hi_bp": 1e4 * bs["diff_hi"], "p_vs_bgk": bs["p_two_sided"],
            })
    return pd.DataFrame(rows).sort_values(["contract", "instruments", "cvar95_bp"])


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--folds", type=str, default="all")
    args = p.parse_args()
    cfg = load_config(args.config or "configs/experiments/e8_real_backtest.yaml")
    out = ensure_dir(cfg.get("out_dir", "tables"))
    cost = CostModel(underlying_bps=float(cfg.get("cost_bps", 2.0)))
    folds_dir = Path("data/folds")
    if args.smoke:
        ensure_smoke_folds(folds_dir)
    dates, spot, surfaces = load_real_market()
    fold_ids = list(range(n_folds(folds_dir))) if args.folds == "all" else [int(x) for x in args.folds.split(",")]

    rows = []
    for cpath in cfg.get("contracts", ["configs/contracts/doc_090.yaml"]):
        contract = load_contract(cpath)
        for instr in cfg.get("instruments", ["I1"]):
            instruments = instrument_set_for(instr, contract.barrier_mult)
            for k in fold_ids:
                fold = load_fold(folds_dir / f"fold_{k}")
                paths, starts = real_windows(dates, spot, surfaces, fold.dates_test, contract.maturity_days)
                runs = discover_runs("models/hedge", Path(cpath).stem, instr, k)
                ppo = discover_runs("models/hedge_ppo", Path(cpath).stem, instr, k)
                res = evaluate_all(paths, contract, instruments, cost, runs, ppo)
                for hedger, seeds in res.items():
                    for seed, L in seeds.items():
                        rows.append(pd.DataFrame({"contract": Path(cpath).stem, "instruments": instr,
                                                  "fold": k, "hedger": hedger, "seed": seed,
                                                  "date": starts, "loss": L}))
                print(f"{Path(cpath).stem} {instr} fold {k}: {len(starts)} real windows, "
                      f"{sum(len(v) for v in runs.values())} deep + {sum(len(v) for v in ppo.values())} ppo runs")
    losses = pd.concat(rows, ignore_index=True)
    # classical baselines appear once per instrument set; keep only folds where every hedger exists
    losses.to_parquet(out / "real_backtest_losses.parquet", index=False)
    summ = summarise(losses, block=int(cfg.get("block", 30)), n_boot=200 if args.smoke else int(cfg.get("n_boot", 2000)))
    summ.to_csv(out / "real_backtest_summary.csv", index=False)
    write_manifest(out / "_e8", config=cfg, seed=args.seed)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    print("\nunits: basis points of notional (loss / S0 * 1e4); lower CVaR is better; "
          "diff < 0 means better than BGK barrier delta.")


if __name__ == "__main__":
    main()
