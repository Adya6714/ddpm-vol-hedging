"""Implied forward from put-call parity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_forward_from_parity(
    strikes: np.ndarray,
    calls: np.ndarray,
    puts: np.ndarray,
    discount: float,
    spot: float | None = None,
    n_near_atm: int = 3,
) -> float:
    """F = K + (C - P) / DF using the n_near_atm strikes closest to spot/ATM."""
    strikes = np.asarray(strikes, dtype=float)
    calls = np.asarray(calls, dtype=float)
    puts = np.asarray(puts, dtype=float)
    mask = np.isfinite(strikes) & np.isfinite(calls) & np.isfinite(puts)
    strikes, calls, puts = strikes[mask], calls[mask], puts[mask]
    if len(strikes) == 0:
        if spot is None:
            raise ValueError("No valid put-call pairs and no spot fallback")
        return float(spot)
    ref = float(spot) if spot is not None else float(np.median(strikes))
    order = np.argsort(np.abs(strikes - ref))[:n_near_atm]
    Fs = strikes[order] + (calls[order] - puts[order]) / max(discount, 1e-12)
    return float(np.median(Fs))


def forwards_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Group option rows by date/expiry and compute implied forwards.

    Expects columns: date, expiry, strike, call, put, df, spot (optional), fut (optional).
    """
    required = {"date", "expiry", "strike", "call", "put", "df"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = []
    for (date, expiry), g in rows.groupby(["date", "expiry"]):
        spot = float(g["spot"].iloc[0]) if "spot" in g.columns else None
        try:
            F = implied_forward_from_parity(
                g["strike"].to_numpy(),
                g["call"].to_numpy(),
                g["put"].to_numpy(),
                float(g["df"].iloc[0]),
                spot=spot,
            )
        except ValueError:
            if "fut" in g.columns and np.isfinite(g["fut"].iloc[0]):
                F = float(g["fut"].iloc[0])
            elif spot is not None:
                F = spot
            else:
                raise
        out.append({"date": date, "expiry": expiry, "forward": F})
    return pd.DataFrame(out)
