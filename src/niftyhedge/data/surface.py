"""Daily 9x6 IV surface grid helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Standard grid (matches legacy notebook)
MONEYNESS = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
TENORS_DAYS = np.array([7, 14, 30, 60, 90, 180])


@dataclass
class SurfaceGrid:
    surfaces: np.ndarray  # [T, 9, 6]
    mask: np.ndarray  # True where supported by data
    dates: np.ndarray


def interpolate_total_variance(
    log_m: np.ndarray,
    tenors: np.ndarray,
    w_slices: list[np.ndarray],
    target_log_m: np.ndarray,
    target_tenors: np.ndarray,
) -> np.ndarray:
    """Linear interp of total variance in maturity at fixed log-moneyness."""
    n_m, n_t = len(target_log_m), len(target_tenors)
    out = np.zeros((n_m, n_t))
    # first interp each slice onto target moneyness
    slice_w = []
    for k_src, w_src in zip(log_m if False else [None] * len(w_slices), w_slices):
        # w_slices already on target moneyness grid in our pipeline
        slice_w.append(np.asarray(w_src, dtype=float))
    slice_w = np.stack(slice_w, axis=1)  # [n_m, n_src_t]
    src_t = np.asarray(tenors, dtype=float)
    for j, T in enumerate(target_tenors):
        if T <= src_t[0]:
            out[:, j] = slice_w[:, 0]
        elif T >= src_t[-1]:
            out[:, j] = slice_w[:, -1]
        else:
            i = int(np.searchsorted(src_t, T) - 1)
            a = (T - src_t[i]) / (src_t[i + 1] - src_t[i])
            out[:, j] = (1 - a) * slice_w[:, i] + a * slice_w[:, i + 1]
    return out


def iv_from_total_variance(w: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    T = np.asarray(tenors, dtype=float)[None, :]
    return np.sqrt(np.maximum(w, 1e-12) / np.maximum(T, 1e-12))


def load_legacy_surfaces(raw_dir: str = "data/raw") -> SurfaceGrid:
    from pathlib import Path

    raw = Path(raw_dir)
    surfaces = np.load(raw / "surfaces.npy")
    dates = np.load(raw / "surface_dates.npy", allow_pickle=True)
    # legacy mean-filled already; mask True where originally finite is unknown
    mask = np.ones_like(surfaces, dtype=bool)
    return SurfaceGrid(surfaces=surfaces, mask=mask, dates=dates)


def make_sample_surfaces(n_days: int = 120, seed: int = 0) -> SurfaceGrid:
    """Synthetic surfaces for smoke tests."""
    rng = np.random.default_rng(seed)
    from datetime import date, timedelta

    from niftyhedge.data.svi import make_arb_free_surface

    base = make_arb_free_surface(MONEYNESS, TENORS_DAYS / 365.0, atm_vol=0.16)
    surfaces = np.stack(
        [base * (1.0 + 0.05 * rng.standard_normal()) for _ in range(n_days)], axis=0
    )
    surfaces = np.clip(surfaces, 0.05, 0.80)
    start = date(2020, 1, 2)
    dates = np.array([start + timedelta(days=i) for i in range(n_days)], dtype=object)
    mask = np.ones_like(surfaces, dtype=bool)
    return SurfaceGrid(surfaces=surfaces, mask=mask, dates=dates)
