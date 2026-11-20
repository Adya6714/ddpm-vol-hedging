#!/usr/bin/env python3
"""Train deep hedgers (and PPO ablation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common, ensure_smoke_folds, load_contract, n_folds, parse_folds, parse_seeds
from niftyhedge.gen.base import Paths, start_states_from_fold
from niftyhedge.gen.gbm import GBMWorld
from niftyhedge.data.splits import load_fold
from niftyhedge.hedge.deep_hedge import train_deep_hedger
from niftyhedge.hedge.instruments import CostModel, instrument_set_for
from niftyhedge.hedge.ppo import train_ppo
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, manifest_matches, write_manifest
from niftyhedge.utils.seed import set_seed


def _load_train_val(world_name: str, fold_id: int, n_train: int, n_val: int, horizon: int,
                    seed: int, smoke: bool) -> tuple[Paths, Paths]:
    """Disjoint train / validation subsets of the world's generated *training* paths.

    (Previously validation was the first n_val rows of the same file, i.e. a
    subset of the training batch.)
    """
    pt = Path(f"data/paths/{world_name}/fold_{fold_id}/paths_train.pt")
    if pt.exists():
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        S, surf, idx = blob["S"].float(), blob["surf"].float(), blob["start_idx"]
    else:
        if not smoke:
            raise FileNotFoundError(f"{pt} missing - run scripts/11_generate_paths.py first")
        fold = load_fold(Path("data/folds") / f"fold_{fold_id}")
        world = GBMWorld()
        world.fit(fold, seed)
        p = world.sample_paths(start_states_from_fold(fold, "train"), n_train + n_val, horizon, seed)
        S, surf, idx = p.S, p.surf, p.start_idx
    n = S.shape[0]
    n_val = min(n_val, n // 5)
    g = torch.Generator().manual_seed(1234 + fold_id)  # same split for every seed
    perm = torch.randperm(n, generator=g)
    va, tr = perm[:n_val], perm[n_val : n_val + n_train]
    return (Paths(S=S[tr], surf=surf[tr], start_idx=idx[tr]),
            Paths(S=S[va], surf=surf[va], start_idx=idx[va]))


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
    fold_ids = parse_folds(args.folds, n=n_folds()) if args.folds else [args.fold if args.fold is not None else 0]
    seeds = parse_seeds(args.seeds)
    horizon = 10 if args.smoke else contract.maturity_days
    n_tr = 64 if args.smoke else int(cfg.get("n_train", 50000))
    n_va = 32 if args.smoke else int(cfg.get("n_val", 10000))

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
            if not args.force and manifest_matches(out, {**cfg, "world": args.world, "fold": k, "instruments": args.instruments}):
                print(f"skip {out}")
                continue
            paths_tr, paths_va = _load_train_val(args.world, k, n_tr, n_va, horizon, seed, args.smoke)
            if cfg.get("algo") == "ppo":
                result = train_ppo(paths_tr, paths_va, contract, cost, seed=seed, out_dir=out, smoke=args.smoke,
                                   iterations=int(cfg.get("iterations", 300)),
                                   batch_size=int(cfg.get("batch_size", 4096)),
                                   ppo_epochs=int(cfg.get("ppo_epochs", 4)), clip=float(cfg.get("clip", 0.2)),
                                   lr=float(cfg.get("lr", 3e-4)), hidden=int(cfg.get("hidden", 128)))
            else:
                result = train_deep_hedger(
                    paths_tr,
                    paths_va,
                    contract,
                    instruments,
                    cost,
                    seed=seed,
                    epochs=int(cfg.get("epochs", 50)),
                    steps=cfg.get("steps"),
                    batch_size=int(cfg.get("batch_size", 4096)),
                    lr=float(cfg.get("lr", 1e-3)),
                    alpha=float(cfg.get("alpha", 0.95)),
                    hidden=int(cfg.get("hidden", 128)),
                    residual=bool(cfg.get("residual", True)),
                    out_dir=out,
                    smoke=args.smoke,
                )
            write_manifest(
                out,
                config={**cfg, "world": args.world, "fold": k, "instruments": args.instruments},
                seed=seed,
                extra={"val_cvar": result["val_cvar"], "action_std": result["action_std"],
                       "bs_val_cvar": result.get("bs_val_cvar"), "contract": Path(args.contract).stem},
            )
            bs = result.get("bs_val_cvar")
            print(f"trained {args.world} {args.instruments} fold={k} seed={seed} "
                  f"val_cvar={result['val_cvar']:.5f} (BS-BGK {bs if bs is None else round(bs, 5)}) "
                  f"act_std={result['action_std']:.4f}  [CVaR95 of loss / S0]")


if __name__ == "__main__":
    main()
