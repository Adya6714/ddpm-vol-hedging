#!/usr/bin/env python3
"""Download NSE bhavcopies (laptop). Use --probe for one date; --smoke for offline stub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import add_common
from niftyhedge.data.download import download_range, parse_date, probe_one_date
from niftyhedge.utils.config import load_config
from niftyhedge.utils.io import write_manifest


def main():
    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
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

    if args.smoke:
        end = start
    log = download_range(
        start,
        end,
        out,
        udiff_from=parse_date(cfg.get("udiff_from", "2024-07-08")),
        udiff_pattern=pattern,
        smoke=args.smoke,
    )
    if args.retry_errors and not args.smoke:
        err = log[log["status"] == "error"]["date"].tolist()
        for d in err:
            download_range(parse_date(d), parse_date(d), out, udiff_pattern=pattern, smoke=False)
    write_manifest(out / "_run", config=cfg, seed=args.seed, extra={"n_ok": int((log.status == "ok").sum())})
    print(log["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
