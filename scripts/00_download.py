#!/usr/bin/env python3
"""Download NSE bhavcopies (laptop). Use --probe for one date; --smoke for offline stub.

Day files land in data/raw/bhav_daily/YYYY-MM-DD.parquet (OOM-safe).
Re-run with the same command to resume. Use --assemble-only after a full pull.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common
from niftyhedge.data.download import (
    assemble_options_parquet,
    download_range,
    parse_date,
    probe_one_date,
)
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--assemble-only", action="store_true", help="Merge bhav_daily/*.parquet only")
    p.add_argument("--no-assemble", action="store_true", help="Skip final merge (faster mid-run)")
    p.add_argument("--no-resume", action="store_true", help="Re-download even if daily file exists")
    args = p.parse_args()
    cfg = load_config(args.config or "configs/data.yaml")
    start = parse_date(args.start or cfg["start"])
    end = parse_date(args.end or cfg["end"])
    out = Path(cfg.get("raw_dir", "data/raw"))
    pattern = cfg.get("udiff_url_pattern")

    if args.probe:
        info = probe_one_date(start, udiff_pattern=pattern)
        print(info)
        if info["status"] != "ok":
            raise SystemExit(1)
        return

    if args.assemble_only:
        assemble_options_parquet(out)
        return

    if args.smoke:
        end = start

    log = download_range(
        start,
        end,
        out,
        udiff_from=parse_date(cfg.get("udiff_from", "2024-07-08")),
        udiff_pattern=pattern,
        smoke=args.smoke,
        resume=not args.no_resume,
        assemble=not args.no_assemble and not args.retry_errors,
        only_errors=False,
    )

    if args.retry_errors and not args.smoke:
        print("retrying error dates...", flush=True)
        log = download_range(
            start,
            end,
            out,
            udiff_from=parse_date(cfg.get("udiff_from", "2024-07-08")),
            udiff_pattern=pattern,
            smoke=False,
            resume=False,
            assemble=not args.no_assemble,
            only_errors=True,
        )

    n_ok = int((log.status == "ok").sum()) if "status" in log.columns else 0
    write_manifest(out / "_run", config=cfg, seed=args.seed, extra={"n_ok": n_ok})
    print(log["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
