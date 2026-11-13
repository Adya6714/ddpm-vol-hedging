#!/usr/bin/env python3
"""Aggregate tables/figures for the paper."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common
from niftyhedge.utils.io import ensure_dir, write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--run", type=str, default="runs/smoke")
    args = p.parse_args()
    run = ensure_dir(args.run)
    for d in ["tables", "figures"]:
        src = Path(d)
        if src.exists():
            dst = ensure_dir(run / d)
            for f in src.glob("*"):
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
    write_manifest(run, config={"run": args.run}, seed=args.seed)
    print(f"copied tables/figures into {run}")


if __name__ == "__main__":
    main()
