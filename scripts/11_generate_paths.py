#!/usr/bin/env python3
"""Generate joint spot+surface paths from fitted worlds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common, ensure_smoke_folds, load_world, parse_folds
from niftyhedge.data.splits import load_fold
from niftyhedge.gen.base import start_states_from_fold
from niftyhedge.utils.io import ensure_dir, write_manifest
from niftyhedge.utils.seed import set_seed


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--worlds", type=str, default="gbm")
    p.add_argument("--folds", type=str, default="0")
    p.add_argument("--n-train", type=int, default=200000)
    p.add_argument("--n-test", type=int, default=100000)
    p.add_argument("--horizon", type=int, default=30)
    args = p.parse_args()
    if args.smoke:
        args.n_train = min(args.n_train, 64)
        args.n_test = min(args.n_test, 32)
        args.horizon = min(args.horizon, 10)
        ensure_smoke_folds(Path("data/folds"))

    worlds = [w.strip() for w in args.worlds.split(",")]
    folds = parse_folds(args.folds, n=len(list(Path("data/folds").glob("fold_*"))) or 1)

    for wname in worlds:
        for k in folds:
            fold = load_fold(Path("data/folds") / f"fold_{k}")
            wpath = Path(f"models/gen/{wname}/fold_{k}/seed_0/world.pkl")
            if not wpath.exists():
                raise FileNotFoundError(f"Missing fitted world {wpath}; run 10_fit_generators first")
            world = load_world(wpath)
            out = ensure_dir(Path(f"data/paths/{wname}/fold_{k}"))
            for split, n in [("train", args.n_train), ("test", args.n_test)]:
                set_seed(args.seed + (0 if split == "train" else 1))
                starts = start_states_from_fold(fold, "train" if split == "train" else "test")
                paths = world.sample_paths(starts, n, args.horizon, seed=args.seed)
                torch.save({"S": paths.S, "surf": paths.surf, "start_idx": paths.start_idx}, out / f"paths_{split}.pt")
            write_manifest(out, config={"world": wname, "fold": k}, seed=args.seed)
            print(f"paths {wname} fold={k} -> {out}")


if __name__ == "__main__":
    main()
