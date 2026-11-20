#!/usr/bin/env python3
"""Export compact JSON for the GitHub Pages site (docs/data/)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)

surf = np.load(root / "data/surfaces/surfaces_v2.npy")
mask = np.load(root / "data/surfaces/surface_mask_v2.npy")
dates = [str(d)[:10] for d in np.load(root / "data/surfaces/surface_dates_v2.npy", allow_pickle=True)]
spot = pd.read_parquet(root / "data/surfaces/spot_v2.parquet").set_index("date").loc[dates, "spot"].to_numpy()
assert surf.shape[0] == len(dates) == len(spot)

json.dump({
    "dates": dates,
    "moneyness": [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20],
    "tenors": [7, 14, 30, 60, 90, 180],
    "spot": [round(float(x), 1) for x in spot],
    # iv in vol points, flattened [day][moneyness*6 + tenor]
    "iv": [[round(float(v) * 100, 2) for v in s.ravel()] for s in surf],
    # quoted-support mask as a 54-char 0/1 string per day
    "mask": ["".join("1" if m else "0" for m in x.ravel()) for x in mask],
}, open(out / "surfaces.json", "w"), separators=(",", ":"))

res = {}
s = pd.read_csv(root / "tables/real_backtest_summary.csv")
res["backtest"] = s[s.contract == "doc_090"].round(2).to_dict("records")
for k in ("I1", "I2"):
    m = pd.read_csv(root / f"tables/cross_world_matrix_{k}.csv").round(1)
    res[f"cross_{k}"] = m.to_dict("records")
res["dataset"] = pd.read_csv(root / "tables/dataset_card.csv").round(4).to_dict("records")[0]
json.dump(res, open(out / "results.json", "w"), separators=(",", ":"))
print({p.name: p.stat().st_size // 1024 for p in out.iterdir()}, "KB")
