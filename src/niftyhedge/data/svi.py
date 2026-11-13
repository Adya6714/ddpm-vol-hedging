"""Raw SVI slice fitting with butterfly constraint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


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
    """Fit raw SVI to total variance. Returns params and RMSE."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[mask], w[mask]
    if len(k) < 3:
        raise ValueError("Need at least 3 points to fit SVI")

    def objective(x: np.ndarray) -> float:
        a, b, rho, m, sigma = x
        if b < 0 or abs(rho) >= 1 or sigma <= 0:
            return 1e12
        if butterfly_constrained and b * (1 + abs(rho)) > 2:
            # Gatheral–Jacquier no-butterfly necessary condition scale
            return 1e12
        pred = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        if np.any(pred <= 0):
            return 1e12
        return float(np.mean((pred - w) ** 2))

    x0 = np.array([float(np.min(w)), 0.1, -0.3, 0.0, 0.1])
    bounds = [(-1.0, 2.0), (1e-6, 5.0), (-0.999, 0.999), (-2.0, 2.0), (1e-4, 2.0)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    p = SVIParams(*map(float, res.x))
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
