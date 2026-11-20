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


_NEEDED = ["trade_date", "instrument", "expiry", "strike", "option_type",
           "close", "settle", "contracts", "underlying"]
_NUMERIC = ["strike", "close", "settle", "contracts", "underlying"]
_FUT_INSTR = ["FUTIDX", "IDF"]
_OPT_INSTR = ["OPTIDX", "IDO"]


def _finalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-cast once, keep NIFTY options + index futures, compact dtypes."""
    df = df[[c for c in _NEEDED if c in df.columns]].copy()
    for c in _NEEDED:
        if c not in df.columns:
            df[c] = ""
    for c in ["instrument", "option_type"]:
        df[c] = df[c].astype(str).str.upper().str.strip()
    is_fut = df["instrument"].isin(_FUT_INSTR)
    is_opt = df["option_type"].isin(["CE", "PE"])
    df = df[is_fut | is_opt].copy()
    df.loc[df["instrument"].isin(_FUT_INSTR), "option_type"] = "FUT"
    for c in _NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ["trade_date", "expiry", "option_type", "instrument"]:
        df[c] = df[c].astype("category")
    return df


def load_nifty_option_rows(raw_dir: str | Path) -> pd.DataFrame:
    """Load NIFTY index option rows *and* index futures rows (memory-safe).

    Reads only the 9 needed columns (the parquet stores everything as strings;
    loading all 16 columns for ~4.4M rows exhausts Colab RAM). Futures rows are
    kept with option_type == "FUT" and are used to anchor forwards.
    """
    raw_dir = Path(raw_dir)
    daily = sorted((raw_dir / "bhav_daily").glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    if daily:
        for i, path in enumerate(daily):
            df = normalize_bhav_df(pq.read_table(path).to_pandas())
            sub = df.loc[df["symbol"].str.upper() == "NIFTY"]
            if len(sub):
                frames.append(_finalize_rows(sub))
            if (i + 1) % 200 == 0:
                print(f"  scanned {i + 1}/{len(daily)} daily files", flush=True)
    else:
        parquet = raw_dir / "nifty_options_raw.parquet"
        if not parquet.exists():
            raise FileNotFoundError(
                f"Need {parquet} or {raw_dir / 'bhav_daily'} — upload Gate 1a outputs to Drive"
            )
        print(f"filtering NIFTY options+futures from {parquet} (9 columns) ...", flush=True)
        dataset = ds.dataset(parquet, format="parquet")
        cols = [c for c in _NEEDED if c in dataset.schema.names]
        table = dataset.to_table(columns=cols, filter=pc.field("symbol") == "NIFTY")
        frames.append(_finalize_rows(table.to_pandas()))
        del table

    if not frames:
        raise RuntimeError("No NIFTY option rows found")
    out = pd.concat(frames, ignore_index=True)
    n_opt = int((out["option_type"] != "FUT").sum())
    print(f"NIFTY option rows: {n_opt:,}  futures rows: {len(out) - n_opt:,}  "
          f"days: {out['trade_date'].nunique()}", flush=True)
    return out


def futures_curve(day: pd.DataFrame) -> dict:
    """{expiry_date: futures settle} for one trade date."""
    fut = day[day["option_type"] == "FUT"]
    out = {}
    for _, row in fut.iterrows():
        px = row["settle"] if np.isfinite(row["settle"]) and row["settle"] > 0 else row["close"]
        if np.isfinite(px) and px > 0:
            try:
                out[_parse_date(row["expiry"])] = float(px)
            except Exception:
                continue
    return out


def _curve_forward(curve: dict, trade_d, exp_d, r: float) -> float | None:
    """Forward for exp_d from the futures curve: exact match, else log-linear carry fit."""
    if not curve:
        return None
    if exp_d in curve:
        return curve[exp_d]
    Ts = np.array([max((e - trade_d).days, 0) / 365.0 for e in curve])
    Fs = np.array(list(curve.values()))
    T = max((exp_d - trade_d).days, 0) / 365.0
    ok = Ts > 0
    if ok.sum() >= 2:
        b, logS = np.polyfit(Ts[ok], np.log(Fs[ok]), 1)
        b = float(np.clip(b, -0.05, 0.20))
    else:
        i = int(np.argmin(Ts))
        b = r
        logS = float(np.log(Fs[i]) - b * Ts[i])
    return float(np.exp(logS + b * T))


def _forward_for_expiry(
    g: pd.DataFrame, r: float, spot: float | None, curve: dict | None = None
) -> float:
    """Forward for one expiry.

    Futures-anchored: the futures curve gives a reference forward; put-call
    parity on the 3 strikes nearest that reference refines it, but is rejected
    if it deviates >1% (thin/stale quotes on far expiries). Without futures,
    falls back to parity around underlying/spot (the old behaviour).
    """
    if curve:
        try:
            td = _parse_date(g["trade_date"].iloc[0])
            ed = _parse_date(g["expiry"].iloc[0])
            F_ref = _curve_forward(curve, td, ed, r)
        except Exception:
            F_ref = None
        if F_ref is not None:
            calls = g[g["option_type"] == "CE"]
            puts = g[g["option_type"] == "PE"]
            merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
            if len(merged):
                T = max((ed - td).days, 1) / 365.0
                try:
                    F_par = implied_forward_from_parity(
                        merged["strike"].to_numpy(dtype=float),
                        merged["close_c"].to_numpy(dtype=float),
                        merged["close_p"].to_numpy(dtype=float),
                        float(np.exp(-r * T)),
                        spot=F_ref,
                    )
                    if np.isfinite(F_par) and abs(F_par / F_ref - 1) < 0.01:
                        return float(F_par)
                except ValueError:
                    pass
            return float(F_ref)
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
    curve = futures_curve(day) if "FUT" in set(day["option_type"].astype(str)) else {}
    day = day[day["option_type"].astype(str).isin(["CE", "PE"])]
    if day.empty:
        return None
    day["strike"] = _to_float(day["strike"])
    day["close"] = _to_float(day["close"])
    day["contracts"] = _to_float(day["contracts"])
    day["settle"] = _to_float(day["settle"]) if "settle" in day.columns else np.nan
    day["underlying"] = _to_float(day["underlying"]) if "underlying" in day.columns else np.nan

    und = day["underlying"].to_numpy()
    und = und[np.isfinite(und) & (und > 0)]
    spot = float(np.median(und)) if len(und) else None

    slices: list[tuple[float, np.ndarray, np.ndarray]] = []  # (T, w_on_grid, k-supported)
    supported = np.zeros((len(MONEYNESS), len(TENORS_DAYS)), dtype=bool)
    if spot is None and curve:
        spot = _curve_forward(curve, trade_d, trade_d, r)  # carry-adjusted spot proxy
    F_atm = spot if spot is not None else float(np.nanmedian(day["strike"]))
    best_gap = 10**9

    for expiry, g in day.groupby("expiry", observed=True):
        g = g.copy()
        try:
            F = _forward_for_expiry(g, r, spot, curve)
        except Exception:
            continue
        if not np.isfinite(F) or F <= 0:
            continue
        dte_e = (_parse_date(expiry) - trade_d).days
        if abs(dte_e - 30) < best_gap:  # report the ~30-day forward
            best_gap, F_atm = abs(dte_e - 30), F
        got = _slice_ivs(g, F, r, trade_d, min_contracts, min_price, min_dte)
        if got is None:
            continue
        k, w, T = got
        k_grid = np.log(MONEYNESS)
        w_grid = None
        try:
            params, _ = fit_svi_slice(k, w, butterfly_constrained=True)
            # judge the fit in implied-vol units (total-variance RMSE is
            # meaningless across tenors: w ~ 4e-4 at 9d vs ~0.01 at 6m)
            iv_rmse = float(np.sqrt(np.mean(
                (np.sqrt(np.maximum(svi_total_variance(k, params), 0) / T) - np.sqrt(w / T)) ** 2)))
            if iv_rmse <= svi_rmse_max:
                w_grid = np.maximum(svi_total_variance(k_grid, params), 1e-8)
        except Exception:
            pass
        if w_grid is None:
            # fallback: linear in w inside the quoted range, flat IV outside
            order = np.argsort(k)
            ks, ws = k[order], w[order]
            w_grid = np.interp(k_grid, ks, ws)
        k_lo, k_hi = float(k.min()), float(k.max())
        tol = 0.01  # one grid point is ~0.05 in log-moneyness
        k_sup = (k_grid >= k_lo - tol) & (k_grid <= k_hi + tol)
        slices.append((T, w_grid, k_sup))

    if len(slices) < 1:
        return None

    slices.sort(key=lambda x: x[0])
    Ts = np.array([s[0] for s in slices])
    W = np.stack([s[1] for s in slices], axis=1)  # [9, n_exp]
    KS = np.stack([s[2] for s in slices], axis=1)  # [9, n_exp] quoted-strike support

    # interpolate total variance onto target tenors; outside the quoted
    # maturities extrapolate with flat implied vol (w proportional to T),
    # not flat total variance (which inflated 7-day vols).
    target_T = TENORS_DAYS / 365.0
    w_out = np.zeros((len(MONEYNESS), len(TENORS_DAYS)))
    mask = np.zeros_like(w_out, dtype=bool)  # True = interpolated from quotes
    for j, T in enumerate(target_T):
        if T <= Ts[0]:
            w_out[:, j] = W[:, 0] * (T / Ts[0])
        elif T >= Ts[-1]:
            w_out[:, j] = W[:, -1] * (T / Ts[-1])
        else:
            i = int(np.searchsorted(Ts, T) - 1)
            a = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
            w_out[:, j] = (1 - a) * W[:, i] + a * W[:, i + 1]
            mask[:, j] = KS[:, i] & KS[:, i + 1]
    # enforce calendar: non-decreasing TV in maturity
    for j in range(1, w_out.shape[1]):
        w_out[:, j] = np.maximum(w_out[:, j], w_out[:, j - 1] * 1.0001)

    iv = np.sqrt(np.maximum(w_out, 1e-12) / target_T[None, :])
    iv = np.clip(iv, 0.03, 1.5)
    if not np.isfinite(iv).all():
        return None
    return iv.astype(np.float64), mask, float(F_atm)


def spot_series_from_raw(raw_dir: str | Path, r: float = 0.065) -> pd.DataFrame:
    """Daily NIFTY spot proxy from the index-futures curve (both bhav formats).

    Legacy bhavcopy has no underlying price, and data/raw/nifty_spot.parquet is
    rounded to 100 and stops in 2024, so spot is taken as the zero-maturity
    intercept of a log-linear fit to the futures curve (carry-adjusted).
    Columns: date, spot, fut1 (front-month settle), fut1_dte.
    """
    parquet = Path(raw_dir) / "nifty_options_raw.parquet"
    dataset = ds.dataset(parquet, format="parquet")
    cols = ["trade_date", "instrument", "expiry", "close", "settle"]
    filt = (pc.field("symbol") == "NIFTY") & pc.field("instrument").isin(_FUT_INSTR)
    df = dataset.to_table(columns=cols, filter=filt).to_pandas()
    for c in ["close", "settle"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["option_type"] = "FUT"
    out = []
    for d, g in df.groupby("trade_date", sort=True):
        td = _parse_date(d)
        curve = futures_curve(g)
        curve = {e: v for e, v in curve.items() if e >= td}
        if not curve:
            continue
        e1 = min(e for e in curve if e >= td)
        out.append({"date": str(td), "spot": _curve_forward(curve, td, td, r),
                    "fut1": curve[e1], "fut1_dte": (e1 - td).days})
    return pd.DataFrame(out)


# bump whenever surface_from_day's output changes; old checkpoints are then ignored
BUILD_VERSION = 2


def _build_one(args):
    d, day, kw = args
    try:
        return str(_parse_date(d)), surface_from_day(day, **kw)
    except Exception as e:  # never let one bad day kill a multi-hour run
        print(f"  WARN {d}: {type(e).__name__}: {e}", flush=True)
        return str(_parse_date(d)), None


def _load_checkpoints(cache_dir: Path) -> dict:
    """{date_str: (iv, mask, F) | None} from all chunk files in cache_dir."""
    done: dict = {}
    for f in sorted(cache_dir.glob("chunk_*.pkl")):
        try:
            done.update(pd.read_pickle(f))
        except Exception as e:  # partial write from a killed session
            print(f"  ignoring unreadable checkpoint {f.name}: {e}", flush=True)
    return done


def _save_chunk(cache_dir: Path, results: dict) -> None:
    if not results:
        return
    keys = sorted(results)
    f = cache_dir / f"chunk_{keys[0]}_{keys[-1]}.pkl"
    tmp = f.with_suffix(".tmp")
    pd.to_pickle(results, tmp)
    tmp.replace(f)  # atomic: a half-written chunk is never picked up


def build_surfaces_from_raw(
    raw_dir: str | Path,
    *,
    r: float = 0.065,
    min_contracts: float = 50.0,
    min_price: float = 1.0,
    min_dte: int = 3,
    svi_rmse_max: float = 0.05,
    cache_dir: str | Path | None = None,
    checkpoint_every: int = 50,
    workers: int = 1,
) -> tuple[SurfaceGrid, pd.DataFrame]:
    """Return SurfaceGrid and forwards table from Gate 1a raw data.

    Resumable: per-day results are saved in chunks under ``cache_dir``
    (point it at Google Drive on Colab). Re-running skips finished days.
    """
    kw = dict(r=r, min_contracts=min_contracts, min_price=min_price,
              min_dte=min_dte, svi_rmse_max=svi_rmse_max)
    done: dict = {}
    if cache_dir is not None:
        import hashlib, json
        key = hashlib.sha1(json.dumps({"v": BUILD_VERSION, **kw}, sort_keys=True).encode()).hexdigest()[:10]
        cache_dir = Path(cache_dir) / f"v{BUILD_VERSION}_{key}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        done = _load_checkpoints(cache_dir)
        if done:
            print(f"resuming: {len(done)} days already in {cache_dir}", flush=True)

    rows = load_nifty_option_rows(raw_dir)
    # one pass groupby instead of a full-table boolean scan per day
    groups = {str(k): g for k, g in rows.groupby("trade_date", observed=True, sort=False)}
    del rows
    todo = [d for d in sorted(groups) if str(_parse_date(d)) not in done]
    print(f"{len(groups)} trade dates, {len(todo)} to build, workers={workers}", flush=True)

    pending: dict = {}

    def _flush():
        if cache_dir is not None:
            _save_chunk(cache_dir, pending)
        done.update(pending)
        pending.clear()

    jobs = ((d, groups[d], kw) for d in todo)
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork" if hasattr(__import__("os"), "fork") else "spawn").Pool(workers)
        it = pool.imap(_build_one, jobs, chunksize=4)
    else:
        pool, it = None, map(_build_one, jobs)
    try:
        for i, (dstr, built) in enumerate(it, 1):
            pending[dstr] = built
            if len(pending) >= checkpoint_every:
                _flush()
            if i % 100 == 0 or i == len(todo):
                ok = sum(v is not None for v in done.values()) + sum(v is not None for v in pending.values())
                print(f"  {i}/{len(todo)} new days processed, {ok} surfaces total", flush=True)
    finally:
        _flush()
        if pool is not None:
            pool.close()
            pool.join()

    surfaces, masks, dates_out, forwards = [], [], [], []
    skipped = 0
    for dstr in sorted(done):
        built = done[dstr]
        if built is None:
            skipped += 1
            continue
        iv, mask, F = built
        surfaces.append(iv)
        masks.append(mask)
        dates_out.append(_parse_date(dstr))
        forwards.append({"date": dstr, "forward": F})

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
