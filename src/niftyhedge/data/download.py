"""NSE bhavcopy download (legacy + UDiFF). Fail loudly; never bare except."""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LEGACY_URL = "https://archives.nseindia.com/content/historical/DERIVATIVES/{year}/{mon}/fo{dd}{mon}{year}bhav.csv.zip"
# Candidate UDiFF pattern — verify with --probe on one known trading day first.
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


def download_range(
    start: date,
    end: date,
    out_dir: str | Path,
    *,
    udiff_from: date = date(2024, 7, 8),
    udiff_pattern: str | None = None,
    smoke: bool = False,
) -> pd.DataFrame:
    """Download weekday bhavcopies. Writes download_log.csv and optional parquet.

    In --smoke mode, writes a tiny synthetic parquet and a log without network I/O.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "download_log.csv"

    if smoke:
        rows = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                rows.append({"date": d.isoformat(), "status": "ok", "message": "smoke", "source": "smoke"})
            d += timedelta(days=1)
        log = pd.DataFrame(rows)
        log.to_csv(log_path, index=False)
        # synthetic option rows
        synth = pd.DataFrame(
            {
                "date": [start.isoformat()] * 4,
                "expiry": [(start + timedelta(days=30)).isoformat()] * 4,
                "strike": [20000, 20000, 21000, 21000],
                "option_type": ["CE", "PE", "CE", "PE"],
                "close": [200.0, 180.0, 120.0, 250.0],
                "volume": [100, 100, 80, 80],
                "spot": [20000.0] * 4,
            }
        )
        synth.to_parquet(out_dir / "nifty_options_raw.parquet", index=False)
        pd.DataFrame({"date": [start.isoformat()], "close": [20000.0]}).to_parquet(
            out_dir / "nifty_spot.parquet", index=False
        )
        pd.DataFrame({"date": [start.isoformat()], "close": [14.0]}).to_parquet(
            out_dir / "india_vix.parquet", index=False
        )
        return log

    sess = _session()
    log_rows = []
    frames = []
    d = start
    while d <= end:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        source = "udiff" if d >= udiff_from else "legacy"
        url = udiff_url(d, udiff_pattern) if source == "udiff" else legacy_url(d)
        try:
            df = fetch_zip_csv(url, sess)
            if df is None:
                log_rows.append({"date": d.isoformat(), "status": "holiday", "message": "404", "source": source})
            else:
                df = df.copy()
                df["trade_date"] = d.isoformat()
                frames.append(df)
                log_rows.append({"date": d.isoformat(), "status": "ok", "message": "", "source": source})
        except Exception as exc:  # noqa: BLE001 — logged explicitly, not silent
            logger.error("Download failed %s: %s", d, exc)
            log_rows.append({"date": d.isoformat(), "status": "error", "message": str(exc), "source": source})
        d += timedelta(days=1)

    log = pd.DataFrame(log_rows)
    log.to_csv(log_path, index=False)
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        raw.to_parquet(out_dir / "nifty_options_raw.parquet", index=False)
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
