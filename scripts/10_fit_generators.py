#!/usr/bin/env python3
"""Fit market generators for given folds/seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common, ensure_smoke_folds, get_world, parse_folds, parse_seeds, save_world
from niftyhedge.data.splits import load_fold
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import ensure_dir, manifest_matches, write_manifest
from niftyhedge.utils.seed import set_seed


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--folds", type=str, default="0")
    p.add_argument("--seeds", type=str, default="0")
    args = p.parse_args()
    cfg = load_config(args.config or "configs/gen/gbm.yaml")
    world_name = cfg["world"]
    folds_dir = Path("data/folds")
    if args.smoke:
        ensure_smoke_folds(folds_dir)
    folds = parse_folds(args.folds, n=len(list(folds_dir.glob("fold_*"))) or 1)
    seeds = parse_seeds(args.seeds)

    for k in folds:
        fold = load_fold(folds_dir / f"fold_{k}")
        for seed in seeds:
            set_seed(seed)
            out = ensure_dir(Path(cfg.get("out_dir", f"models/gen/{world_name}")) / f"fold_{k}" / f"seed_{seed}")
            if not args.force and manifest_matches(out, cfg):
                print(f"skip {out}")
                continue
            world = get_world(world_name, cfg, smoke=args.smoke)
            world.fit(fold, seed)
            save_world(world, out / "world.pkl")
            if hasattr(world, "save"):
                world.save(out / "weights.pt")
            write_manifest(out, config=cfg, seed=seed, extra={"fold": k, "world": world_name})
            print(f"fitted {world_name} fold={k} seed={seed} -> {out}")


if __name__ == "__main__":
    main()
