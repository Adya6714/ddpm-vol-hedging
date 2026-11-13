#!/usr/bin/env python3
"""Train deep hedgers (and PPO ablation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common, ensure_smoke_folds, load_contract, parse_folds, parse_seeds
from niftyhedge.gen.base import Paths, start_states_from_fold
from niftyhedge.gen.gbm import GBMWorld
from niftyhedge.data.splits import load_fold
from niftyhedge.hedge.deep_hedge import train_deep_hedger
from niftyhedge.hedge.instruments import CostModel, instrument_set_for
from niftyhedge.hedge.ppo import train_ppo
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, manifest_matches, write_manifest
from niftyhedge.utils.seed import set_seed


def _load_or_sample_paths(world_name: str, fold_id: int, n: int, horizon: int, seed: int, smoke: bool) -> Paths:
    pt = Path(f"data/paths/{world_name}/fold_{fold_id}/paths_train.pt")
    if pt.exists():
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        S, surf, idx = blob["S"][:n], blob["surf"][:n], blob["start_idx"][:n]
        return Paths(S=S, surf=surf, start_idx=idx)
    # fallback: sample from GBM
    fold = load_fold(Path("data/folds") / f"fold_{fold_id}")
    world = GBMWorld()
    world.fit(fold, seed)
    starts = start_states_from_fold(fold, "train")
    return world.sample_paths(starts, n, horizon, seed)


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--world", type=str, default="gbm")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--folds", type=str, default=None)
    p.add_argument("--instruments", type=str, default="I1")
    p.add_argument("--contract", type=str, default="configs/contracts/doc_090.yaml")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--cost-bps", type=float, default=None)
    args = p.parse_args()
    cfg = load_config(args.config or "configs/hedge/deep_hedge.yaml")
    if args.smoke:
        ensure_smoke_folds(Path("data/folds"))
    contract = load_contract(args.contract)
    instruments = instrument_set_for(args.instruments, contract.barrier_mult)
    cost = CostModel(
        underlying_bps=float(args.cost_bps if args.cost_bps is not None else cfg.get("cost_bps", 2.0)),
        option_frac=float(cfg.get("option_frac", 0.01)),
    )
    fold_ids = parse_folds(args.folds, n=6) if args.folds else [args.fold if args.fold is not None else 0]
    seeds = parse_seeds(args.seeds)
    horizon = 10 if args.smoke else 30
    n_tr = 64 if args.smoke else 4096
    n_va = 32 if args.smoke else 1024

    for k in fold_ids:
        for seed in seeds:
            set_seed(seed)
            out = ensure_dir(
                Path(cfg.get("out_dir", "models/hedge"))
                / args.world
                / args.instruments
                / Path(args.contract).stem
                / f"fold_{k}"
                / f"seed_{seed}"
            )
            if not args.force and manifest_matches(out, {**cfg, "world": args.world, "fold": k}):
                print(f"skip {out}")
                continue
            paths_tr = _load_or_sample_paths(args.world, k, n_tr, horizon, seed, args.smoke)
            paths_va = _load_or_sample_paths(args.world, k, n_va, horizon, seed + 1, args.smoke)
            if cfg.get("algo") == "ppo":
                result = train_ppo(paths_tr, paths_va, contract, cost, seed=seed, out_dir=out, smoke=args.smoke)
            else:
                result = train_deep_hedger(
                    paths_tr,
                    paths_va,
                    contract,
                    instruments,
                    cost,
                    seed=seed,
                    epochs=int(cfg.get("epochs", 50)),
                    out_dir=out,
                    smoke=args.smoke,
                )
            write_manifest(
                out,
                config={**cfg, "world": args.world, "fold": k, "instruments": args.instruments},
                seed=seed,
                extra={"val_cvar": result["val_cvar"], "action_std": result["action_std"]},
            )
            print(f"trained {args.world} fold={k} seed={seed} val_cvar={result['val_cvar']:.4f} act_std={result['action_std']:.4f}")


if __name__ == "__main__":
    main()
