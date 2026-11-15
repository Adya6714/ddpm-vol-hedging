"""NSE bhavcopy download (legacy + UDiFF). Fail loudly; never bare except.

Writes one parquet per day under data/raw/bhav_daily/ so a full 2018–2026
pull cannot OOM. Use assemble_options_parquet() to build the combined file.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)

LEGACY_URL = "https://archives.nseindia.com/content/historical/DERIVATIVES/{year}/{mon}/fo{dd}{mon}{year}bhav.csv.zip"
UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; niftyhedge/0.2; research)",
    "Accept": "*/*",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def legacy_url(d: date) -> str:
    return LEGACY_URL.format(year=d.year, mon=d.strftime("%b").upper(), dd=d.strftime("%d"))


def udiff_url(d: date, pattern: str | None = None) -> str:
    pat = pattern or UDIFF_URL
    return pat.format(yyyymmdd=d.strftime("%Y%m%d"), YYYYMMDD=d.strftime("%Y%m%d"))


def fetch_zip_csv(url: str, session: requests.Session | None = None, timeout: int = 60) -> pd.DataFrame | None:
    """Download a zip containing one CSV. Returns None on 404 (holiday). Raises on other errors."""
    sess = session or _session()
    resp = sess.get(url, timeout=timeout)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {url}")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV in zip from {url}")
        with zf.open(names[0]) as f:
            return pd.read_csv(f)


def _daily_path(out_dir: Path, d: date) -> Path:
    return out_dir / "bhav_daily" / f"{d.isoformat()}.parquet"


def _load_log(log_path: Path) -> dict[str, dict]:
    if not log_path.exists():
        return {}
    df = pd.read_csv(log_path)
    return {str(r["date"]): r.to_dict() for _, r in df.iterrows()}


def _save_log(log_path: Path, by_date: dict[str, dict]) -> pd.DataFrame:
    rows = [by_date[k] for k in sorted(by_date)]
    log = pd.DataFrame(rows)
    log.to_csv(log_path, index=False)
    return log


# Canonical columns after normalizing legacy + UDiFF bhavcopies.
CANONICAL_COLS = [
    "trade_date",
    "instrument",
    "symbol",
    "expiry",
    "strike",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "contracts",
    "value",
    "open_int",
    "chg_oi",
    "underlying",
]

_LEGACY_MAP = {
    "trade_date": "trade_date",
    "INSTRUMENT": "instrument",
    "SYMBOL": "symbol",
    "EXPIRY_DT": "expiry",
    "STRIKE_PR": "strike",
    "OPTION_TYP": "option_type",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "SETTLE_PR": "settle",
    "CONTRACTS": "contracts",
    "VAL_INLAKH": "value",
    "OPEN_INT": "open_int",
    "CHG_IN_OI": "chg_oi",
}

_UDIFF_MAP = {
    "trade_date": "trade_date",
    "FinInstrmTp": "instrument",
    "TckrSymb": "symbol",
    "XpryDt": "expiry",
    "StrkPric": "strike",
    "OptnTp": "option_type",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "SttlmPric": "settle",
    "TtlTradgVol": "contracts",
    "TtlTrfVal": "value",
    "OpnIntrst": "open_int",
    "ChngInOpnIntrst": "chg_oi",
    "UndrlygPric": "underlying",
}


def normalize_bhav_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy or UDiFF columns onto a single canonical schema (all strings)."""
    cols = set(df.columns)
    if "SYMBOL" in cols:
        mapping = _LEGACY_MAP
    elif "TckrSymb" in cols:
        mapping = _UDIFF_MAP
    else:
        raise ValueError(f"Unrecognized bhav schema: {sorted(cols)[:20]}")
    out = pd.DataFrame(index=df.index)
    for src, dst in mapping.items():
        if src in df.columns:
            out[dst] = df[src].where(pd.notna(df[src]), "").astype(str)
    for c in CANONICAL_COLS:
        if c not in out.columns:
            out[c] = ""
        else:
            out[c] = out[c].where(pd.notna(out[c]), "").astype(str)
    return out[CANONICAL_COLS]


def _canonical_table(df: pd.DataFrame) -> pa.Table:
    norm = normalize_bhav_df(df)
    arrays = [pa.array(norm[c].to_numpy(dtype=object), type=pa.large_string()) for c in CANONICAL_COLS]
    return pa.Table.from_arrays(arrays, names=CANONICAL_COLS)


def assemble_options_parquet(out_dir: str | Path, dest_name: str = "nifty_options_raw.parquet") -> Path:
    """Stream-merge daily parquets into one canonical file without holding all days in RAM."""
    out_dir = Path(out_dir)
    daily = sorted((out_dir / "bhav_daily").glob("*.parquet"))
    if not daily:
        raise FileNotFoundError(f"No daily files in {out_dir / 'bhav_daily'}")
    dest = out_dir / dest_name
    tmp = out_dir / (dest_name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    schema = pa.schema([(c, pa.large_string()) for c in CANONICAL_COLS])
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    n_rows = 0
    try:
        for i, path in enumerate(daily):
            df = pq.read_table(path).to_pandas()
            table = _canonical_table(df)
            writer.write_table(table)
            n_rows += table.num_rows
            if (i + 1) % 100 == 0:
                print(f"  assembled {i + 1}/{len(daily)} days ({n_rows:,} rows)", flush=True)
    finally:
        writer.close()
    tmp.replace(dest)
    print(f"wrote {dest} ({n_rows:,} rows from {len(daily)} days)", flush=True)
    return dest


def download_range(
    start: date,
    end: date,
    out_dir: str | Path,
    *,
    udiff_from: date = date(2024, 7, 8),
    udiff_pattern: str | None = None,
    smoke: bool = False,
    resume: bool = True,
    assemble: bool = True,
    only_errors: bool = False,
) -> pd.DataFrame:
    """Download weekday bhavcopies day-by-day to disk.

    resume=True skips dates that already have status=ok and a daily parquet.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = out_dir / "bhav_daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "download_log.csv"

    if smoke:
        rows = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                rows.append({"date": d.isoformat(), "status": "ok", "message": "smoke", "source": "smoke"})
                pd.DataFrame(
                    {
                        "trade_date": [d.isoformat()] * 2,
                        "SYMBOL": ["NIFTY", "NIFTY"],
                        "strike": [20000, 20000],
                        "option_type": ["CE", "PE"],
                        "close": [200.0, 180.0],
                        "volume": [100, 100],
                    }
                ).to_parquet(_daily_path(out_dir, d), index=False)
            d += timedelta(days=1)
        log = pd.DataFrame(rows)
        log.to_csv(log_path, index=False)
        assemble_options_parquet(out_dir)
        pd.DataFrame({"date": [start.isoformat()], "close": [20000.0]}).to_parquet(
            out_dir / "nifty_spot.parquet", index=False
        )
        pd.DataFrame({"date": [start.isoformat()], "close": [14.0]}).to_parquet(
            out_dir / "india_vix.parquet", index=False
        )
        return log

    by_date = _load_log(log_path)
    sess = _session()
    done = 0
    fetched = 0
    d = start
    while d <= end:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        key = d.isoformat()
        existing = by_date.get(key)
        day_file = _daily_path(out_dir, d)

        if only_errors:
            if existing is None or existing.get("status") != "error":
                d += timedelta(days=1)
                continue
        elif resume and existing is not None and existing.get("status") == "ok" and day_file.exists():
            d += timedelta(days=1)
            done += 1
            continue

        source = "udiff" if d >= udiff_from else "legacy"
        url = udiff_url(d, udiff_pattern) if source == "udiff" else legacy_url(d)
        try:
            df = fetch_zip_csv(url, sess)
            if df is None:
                by_date[key] = {"date": key, "status": "holiday", "message": "404", "source": source}
                if day_file.exists():
                    day_file.unlink()
            else:
                df = df.copy()
                df["trade_date"] = key
                # cast object cols to string for stable parquet schema across days
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str)
                df.to_parquet(day_file, index=False)
                by_date[key] = {"date": key, "status": "ok", "message": "", "source": source}
                fetched += 1
        except Exception as exc:  # noqa: BLE001 — logged explicitly, not silent
            logger.error("Download failed %s: %s", d, exc)
            by_date[key] = {"date": key, "status": "error", "message": str(exc), "source": source}

        done += 1
        if done % 25 == 0:
            _save_log(log_path, by_date)
            print(f"  progress through {key}: fetched={fetched} skipped_ok={done - fetched}", flush=True)
        d += timedelta(days=1)

    log = _save_log(log_path, by_date)
    print(log["status"].value_counts().to_dict(), flush=True)
    if assemble:
        # only assemble if we have daily files
        if any(daily_dir.glob("*.parquet")):
            assemble_options_parquet(out_dir)
        else:
            print("no daily parquets yet — skip assemble", flush=True)
    return log


def probe_one_date(d: date, udiff_pattern: str | None = None) -> dict:
    """Test URL pattern for a single date. Returns status dict."""
    sess = _session()
    url = udiff_url(d, udiff_pattern) if d >= date(2024, 7, 8) else legacy_url(d)
    try:
        df = fetch_zip_csv(url, sess)
        if df is None:
            return {"date": d.isoformat(), "status": "holiday_or_404", "url": url, "rows": 0}
        return {"date": d.isoformat(), "status": "ok", "url": url, "rows": len(df)}
    except Exception as exc:  # noqa: BLE001
        return {"date": d.isoformat(), "status": "error", "url": url, "message": str(exc)}


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()
