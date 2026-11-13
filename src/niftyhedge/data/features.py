"""Leak-free conditioning features (lag >= 1)."""

from __future__ import annotations

import numpy as np

from niftyhedge.data.events import event_flag_series
from niftyhedge.utils.io import assert_finite


def build_conditioning(
    surfaces: np.ndarray,
    dates,
    spot: np.ndarray,
    vix: np.ndarray,
    events: list[int] | None = None,
    lag: int = 1,
    ewma_alpha: float = 0.4,
) -> np.ndarray:
    """Build conditioning rows so row t uses data dated <= t - lag only.

    Events for day t may use a pre-announced calendar (known before t).
    Returns array [T, D] with D = 54 (lagged surface) + 54 (EWMA) + 3 (ret, ret2, vix) + 1 (event).
    """
    if lag < 1:
        raise ValueError("lag must be >= 1 to avoid target leakage")
    surfaces = np.asarray(surfaces, dtype=float)
    spot = np.asarray(spot, dtype=float)
    vix = np.asarray(vix, dtype=float)
    T = surfaces.shape[0]
    if surfaces.ndim != 3:
        raise ValueError("surfaces must be [T, n_m, n_tau]")
    if len(spot) != T or len(vix) != T or len(dates) != T:
        raise ValueError("surfaces/spot/vix/dates length mismatch")

    if events is None:
        events = event_flag_series(dates)
    events = list(events)
    if len(events) != T:
        raise ValueError("events length mismatch")

    flat = surfaces.reshape(T, -1)
    ewma = np.zeros_like(flat)
    # EWMA of past surfaces only: ewma[t] uses surfaces[t-lag] and earlier
    for t in range(T):
        src = t - lag
        if src < 0:
            ewma[t] = 0.0
        elif src == 0:
            ewma[t] = flat[0]
        else:
            # recursive EWMA ending at src (no same-day surface)
            prev = ewma[t - 1] if t > 0 else flat[0]
            # rebuild from scratch up to src for clarity / correctness
            cur = flat[0].copy()
            for i in range(1, src + 1):
                cur = ewma_alpha * flat[i] + (1.0 - ewma_alpha) * cur
            ewma[t] = cur

    ret = np.zeros(T)
    ret[1:] = np.log(spot[1:] / spot[:-1])
    lagged_ret = np.zeros(T)
    lagged_ret[lag:] = ret[lag:]
    # use return ending at t-lag: ret[t-lag] = log(S[t-lag]/S[t-lag-1])
    lagged_ret = np.zeros(T)
    for t in range(T):
        i = t - lag
        if i >= 1:
            lagged_ret[t] = ret[i]

    lagged_vix = np.zeros(T)
    for t in range(T):
        i = t - lag
        if i >= 0:
            lagged_vix[t] = vix[i]

    lagged_surf = np.zeros_like(flat)
    for t in range(T):
        i = t - lag
        if i >= 0:
            lagged_surf[t] = flat[i]

    # event for day t is allowed (pre-announced)
    ev = np.asarray(events, dtype=float).reshape(T, 1)
    C = np.concatenate(
        [
            lagged_surf,
            ewma,
            lagged_ret.reshape(T, 1),
            (lagged_ret**2).reshape(T, 1),
            lagged_vix.reshape(T, 1),
            ev,
        ],
        axis=1,
    )
    assert_finite(C[lag:], "conditioning[lag:]")
    return C


def atm_30d(surfaces: np.ndarray) -> np.ndarray:
    """ATM is moneyness index 4, 30d is tenor index 2 on the 9x6 grid."""
    return np.asarray(surfaces)[:, 4, 2]
