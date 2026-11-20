"""Raw SVI slice fitting with butterfly constraint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))


def fit_svi_slice(
    k: np.ndarray,
    w: np.ndarray,
    *,
    butterfly_constrained: bool = True,
) -> tuple[SVIParams, float]:
    """Fit raw SVI to total variance. Returns params and RMSE (total-variance units).

    The fit is done on w / median(w): short-dated slices have w ~ 1e-4, and an
    unscaled optimiser stops immediately (the old version returned ~40% ATM vol
    for 9-day NIFTY slices whose market ATM was 13%). Several starts, bounded
    least squares; the best fit is kept.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[mask], w[mask]
    if len(k) < 3:
        raise ValueError("Need at least 3 points to fit SVI")

    scale = float(np.median(w))
    y = w / scale
    k_atm = float(k[np.argmin(np.abs(k))])

    def resid(x: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = x
        pred = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        r = pred - y
        # soft constraints: positive minimum variance, Roger Lee wing bound
        min_w = a + b * sigma * np.sqrt(max(1 - rho**2, 0.0))
        pen = [10.0 * max(0.0, -min_w)]
        if butterfly_constrained:
            pen.append(10.0 * max(0.0, b * scale * (1 + abs(rho)) - 2.0))
        return np.concatenate([r, pen])

    lo = [-5.0, 0.0, -0.999, -1.0, 1e-3]
    hi = [10.0, 200.0, 0.999, 1.0, 2.0]
    best = None
    for rho0 in (-0.6, -0.2, 0.2):
        for sig0 in (0.05, 0.2):
            x0 = np.array([float(y.min()) * 0.5, 1.0, rho0, k_atm, sig0])
            x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            try:
                res = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=2000)
            except Exception:
                continue
            if best is None or res.cost < best.cost:
                best = res
    if best is None:
        raise RuntimeError("SVI fit failed")
    a, b, rho, m, sigma = best.x
    p = SVIParams(a=float(a * scale), b=float(b * scale), rho=float(rho), m=float(m), sigma=float(sigma))
    pred = svi_total_variance(k, p)
    rmse = float(np.sqrt(np.mean((pred - w) ** 2)))
    return p, rmse


def make_arb_free_surface(
    moneyness: np.ndarray,
    tenors: np.ndarray,
    atm_vol: float = 0.16,
) -> np.ndarray:
    """Synthetic arbitrage-free surface: flat ATM + mild skew, increasing TV."""
    n_m, n_t = len(moneyness), len(tenors)
    surf = np.zeros((n_m, n_t))
    for j, T in enumerate(tenors):
        T = max(float(T), 1e-6)
        for i, m in enumerate(moneyness):
            k = np.log(m)
            # mild skew + increasing total variance in T
            w = (atm_vol**2) * T * (1.0 + 0.05 * k + 0.02 * k**2)
            surf[i, j] = np.sqrt(max(w, 1e-8) / T)
    return surf
