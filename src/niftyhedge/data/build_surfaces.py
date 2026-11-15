"""Build daily 9x6 IV surfaces from NSE bhav (legacy + UDiFF), SVI-smoothed."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from niftyhedge.data.download import normalize_bhav_df
from niftyhedge.data.forwards import implied_forward_from_parity
from niftyhedge.data.iv import implied_vol
from niftyhedge.data.surface import MONEYNESS, TENORS_DAYS, SurfaceGrid
from niftyhedge.data.svi import fit_svi_slice, svi_total_variance


def _to_float(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _parse_date(x):
    """Parse trade/expiry dates from ISO or legacy NSE forms like 28-Mar-2018."""
    if hasattr(x, "date") and not isinstance(x, str):
        return x.date() if hasattr(x, "hour") else x
    s = str(x).strip()
    # ISO date prefix only when it really is YYYY-MM-DD
    if len(s) >= 10 and s[4] == "-" and s[7] == "-" and s[:4].isdigit():
        s = s[:10]
    return pd.Timestamp(s).date()


def load_nifty_option_rows(raw_dir: str | Path) -> pd.DataFrame:
    """Load NIFTY index option rows only (memory-safe).

    Prefers data/raw/bhav_daily/*.parquet; else filters nifty_options_raw.parquet.
    """
    raw_dir = Path(raw_dir)
    daily = sorted((raw_dir / "bhav_daily").glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    if daily:
        for i, path in enumerate(daily):
            df = normalize_bhav_df(pq.read_table(path).to_pandas())
            m = (df["symbol"].str.upper() == "NIFTY") & (
                df["instrument"].str.upper().isin(["OPTIDX", "IDO"])
            )
            sub = df.loc[m]
            if len(sub):
                frames.append(sub)
            if (i + 1) % 200 == 0:
                print(f"  scanned {i + 1}/{len(daily)} daily files", flush=True)
    else:
        parquet = raw_dir / "nifty_options_raw.parquet"
        if not parquet.exists():
            raise FileNotFoundError(
                f"Need {parquet} or {raw_dir / 'bhav_daily'} — upload Gate 1a outputs to Drive"
            )
        print(f"filtering NIFTY options from {parquet} ...", flush=True)
        dataset = ds.dataset(parquet, format="parquet")
        # canonical schema uses 'symbol'
        table = dataset.to_table(filter=pc.field("symbol") == "NIFTY")
        df = table.to_pandas()
        # already canonical if assembled with normalize; else try normalize
        if "INSTRUMENT" in df.columns or "TckrSymb" in df.columns:
            df = normalize_bhav_df(df)
        inst = df["instrument"].str.upper().str.strip()
        m = inst.isin(["OPTIDX", "IDO"]) | df["option_type"].str.upper().str.strip().isin(["CE", "PE"])
        frames.append(df.loc[m & (df["symbol"].str.upper() == "NIFTY")])

    if not frames:
        raise RuntimeError("No NIFTY option rows found")
    out = pd.concat(frames, ignore_index=True)
    out["option_type"] = out["option_type"].str.upper().str.strip()
    out = out[out["option_type"].isin(["CE", "PE"])]
    print(f"NIFTY option rows: {len(out):,}  days: {out['trade_date'].nunique()}", flush=True)
    return out


def _forward_for_expiry(g: pd.DataFrame, r: float, spot: float | None) -> float:
    """Put-call parity forward; fallback to underlying or spot."""
    calls = g[g["option_type"] == "CE"]
    puts = g[g["option_type"] == "PE"]
    if len(calls) and len(puts):
        merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
        if len(merged):
            # time to expiry in years from first row
            try:
                td = _parse_date(g["trade_date"].iloc[0])
                ed = _parse_date(g["expiry"].iloc[0])
                T = max((ed - td).days, 1) / 365.0
            except Exception:
                T = 30 / 365.0
            dfact = float(np.exp(-r * T))
            und = None
            if "underlying" in g.columns:
                u = _to_float(g["underlying"])
                u = u[np.isfinite(u) & (u > 0)]
                if len(u):
                    und = float(np.median(u))
            return implied_forward_from_parity(
                merged["strike"].to_numpy(dtype=float),
                merged["close_c"].to_numpy(dtype=float),
                merged["close_p"].to_numpy(dtype=float),
                dfact,
                spot=und if und is not None else spot,
            )
    und = _to_float(g["underlying"]) if "underlying" in g.columns else np.array([])
    und = und[np.isfinite(und) & (und > 0)]
    if len(und):
        return float(np.median(und))
    if spot is not None:
        return float(spot)
    # last resort: median strike
    return float(np.nanmedian(_to_float(g["strike"])))


def _slice_ivs(
    g: pd.DataFrame,
    F: float,
    r: float,
    trade_d,
    min_contracts: float,
    min_price: float,
    min_dte: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return log-moneyness, total variance, T_years for one expiry (OTM only)."""
    exp = _parse_date(g["expiry"].iloc[0])
    dte = (exp - trade_d).days
    if dte < min_dte:
        return None
    T = dte / 365.0
    strike = _to_float(g["strike"])
    px = _to_float(g["close"])
    # prefer settle if close missing
    settle = _to_float(g["settle"]) if "settle" in g.columns else np.full(len(g), np.nan)
    px = np.where(np.isfinite(px) & (px > 0), px, settle)
    vol = _to_float(g["contracts"]) if "contracts" in g.columns else np.zeros(len(g))
    ot = g["option_type"].to_numpy()

    # OTM: put if K < F, call if K >= F
    keep = []
    kinds = []
    for i in range(len(g)):
        if not np.isfinite(strike[i]) or not np.isfinite(px[i]):
            continue
        if px[i] < min_price:
            continue
        if np.isfinite(vol[i]) and vol[i] < min_contracts:
            continue
        if strike[i] < F and ot[i] != "PE":
            continue
        if strike[i] >= F and ot[i] != "CE":
            continue
        keep.append(i)
        kinds.append("put" if ot[i] == "PE" else "call")
    if len(keep) < 3:
        return None
    keep = np.asarray(keep)
    K = strike[keep]
    P = px[keep]
    kind = np.asarray(kinds)
    # IV
    ivs = []
    ks = []
    for i in range(len(K)):
        # Black-76 style: underlier = forward, rate = 0
        iv = implied_vol(
            np.array([P[i]]),
            np.array([F]),
            np.array([K[i]]),
            np.array([T]),
            0.0,
            option=kind[i],
        )
        val = float(np.asarray(iv).ravel()[0])
        if np.isfinite(val) and 0.02 < val < 2.5:
            ivs.append(val)
            ks.append(np.log(K[i] / F))
    if len(ivs) < 3:
        return None
    k = np.asarray(ks)
    w = (np.asarray(ivs) ** 2) * T
    return k, w, T


def surface_from_day(
    day: pd.DataFrame,
    *,
    r: float = 0.065,
    min_contracts: float = 50,
    min_price: float = 1.0,
    min_dte: int = 3,
    svi_rmse_max: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Build one 9x6 surface + support mask + ATM forward for a trade date."""
    trade_d = _parse_date(day["trade_date"].iloc[0])
    day = day.copy()
    day["strike"] = _to_float(day["strike"])
    day["close"] = _to_float(day["close"])
    day["contracts"] = _to_float(day["contracts"])
    day["settle"] = _to_float(day["settle"]) if "settle" in day.columns else np.nan
    day["underlying"] = _to_float(day["underlying"]) if "underlying" in day.columns else np.nan

    und = day["underlying"].to_numpy()
    und = und[np.isfinite(und) & (und > 0)]
    spot = float(np.median(und)) if len(und) else None

    slices: list[tuple[float, np.ndarray]] = []  # (T, w_on_grid)
    supported = np.zeros((len(MONEYNESS), len(TENORS_DAYS)), dtype=bool)
    F_atm = spot if spot is not None else float(np.nanmedian(day["strike"]))

    for expiry, g in day.groupby("expiry"):
        g = g.copy()
        try:
            F = _forward_for_expiry(g, r, spot)
        except Exception:
            continue
        if not np.isfinite(F) or F <= 0:
            continue
        F_atm = F
        got = _slice_ivs(g, F, r, trade_d, min_contracts, min_price, min_dte)
        if got is None:
            continue
        k, w, T = got
        try:
            params, rmse = fit_svi_slice(k, w, butterfly_constrained=True)
            if rmse > svi_rmse_max:
                # still use fit if reasonable; else skip
                if rmse > 0.15:
                    continue
            k_grid = np.log(MONEYNESS)
            w_grid = svi_total_variance(k_grid, params)
            w_grid = np.maximum(w_grid, 1e-8)
        except Exception:
            # fallback: interpolate raw w in k
            order = np.argsort(k)
            k_grid = np.log(MONEYNESS)
            w_grid = np.interp(k_grid, k[order], w[order], left=np.nan, right=np.nan)
            if not np.isfinite(w_grid).any():
                continue
            # fill ends
            finite = np.isfinite(w_grid)
            if not finite.any():
                continue
            first, last = np.where(finite)[0][[0, -1]]
            w_grid[:first] = w_grid[first]
            w_grid[last + 1 :] = w_grid[last]
            w_grid = np.where(np.isfinite(w_grid), w_grid, np.nanmean(w_grid))
        slices.append((T, w_grid))

    if len(slices) < 1:
        return None

    slices.sort(key=lambda x: x[0])
    Ts = np.array([s[0] for s in slices])
    W = np.stack([s[1] for s in slices], axis=1)  # [9, n_exp]

    # interpolate total variance onto target tenors
    target_T = TENORS_DAYS / 365.0
    w_out = np.zeros((len(MONEYNESS), len(TENORS_DAYS)))
    mask = np.zeros_like(w_out, dtype=bool)
    for j, T in enumerate(target_T):
        if T <= Ts[0]:
            w_out[:, j] = W[:, 0]
            mask[:, j] = False  # extrapolated
        elif T >= Ts[-1]:
            w_out[:, j] = W[:, -1]
            mask[:, j] = False
        else:
            i = int(np.searchsorted(Ts, T) - 1)
            a = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
            w_out[:, j] = (1 - a) * W[:, i] + a * W[:, i + 1]
            mask[:, j] = True
        # mark moneyness support roughly within observed k range of nearest slice
        mask[:, j] = mask[:, j]  # already
    # enforce calendar: non-decreasing TV in maturity
    for j in range(1, w_out.shape[1]):
        w_out[:, j] = np.maximum(w_out[:, j], w_out[:, j - 1] * 1.0001)

    iv = np.sqrt(np.maximum(w_out, 1e-12) / target_T[None, :])
    iv = np.clip(iv, 0.03, 1.5)
    if not np.isfinite(iv).all():
        return None
    return iv.astype(np.float64), mask, float(F_atm)


def build_surfaces_from_raw(
    raw_dir: str | Path,
    *,
    r: float = 0.065,
    min_contracts: float = 50.0,
    min_price: float = 1.0,
    min_dte: int = 3,
    svi_rmse_max: float = 0.05,
) -> tuple[SurfaceGrid, pd.DataFrame]:
    """Return SurfaceGrid and forwards table from Gate 1a raw data."""
    rows = load_nifty_option_rows(raw_dir)
    dates_sorted = sorted(rows["trade_date"].unique(), key=lambda x: str(x))
    surfaces = []
    masks = []
    dates_out = []
    forwards = []
    skipped = 0
    for i, d in enumerate(dates_sorted):
        day = rows[rows["trade_date"] == d]
        built = surface_from_day(
            day,
            r=r,
            min_contracts=min_contracts,
            min_price=min_price,
            min_dte=min_dte,
            svi_rmse_max=svi_rmse_max,
        )
        if built is None:
            skipped += 1
            continue
        iv, mask, F = built
        surfaces.append(iv)
        masks.append(mask)
        dates_out.append(_parse_date(d))
        forwards.append({"date": str(_parse_date(d)), "forward": F})
        if (i + 1) % 100 == 0:
            print(f"  surfaces {len(surfaces)} built / {i + 1} days scanned (skipped {skipped})", flush=True)

    if not surfaces:
        raise RuntimeError("No surfaces could be built from raw options")
    grid = SurfaceGrid(
        surfaces=np.stack(surfaces, axis=0),
        mask=np.stack(masks, axis=0),
        dates=np.array(dates_out, dtype=object),
    )
    print(
        f"built {len(surfaces)} surfaces, skipped {skipped} days, "
        f"range {dates_out[0]} → {dates_out[-1]}",
        flush=True,
    )
    return grid, pd.DataFrame(forwards)
