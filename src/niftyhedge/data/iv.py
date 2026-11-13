"""Black–Scholes pricing and implied volatility."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS = 1e-12


def bs_price(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
    option: str = "call",
) -> np.ndarray:
    """Vectorised Black–Scholes price. T in years, sigma absolute."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.maximum(T, _EPS)
    sigma = np.maximum(sigma, _EPS)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError(f"option must be call/put, got {option}")


def bs_vega(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.maximum(T, _EPS)
    sigma = np.maximum(sigma, _EPS)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(
    price: np.ndarray | float,
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    option: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> np.ndarray:
    """Newton implied vol; returns NaN where price is outside bounds."""
    price = np.atleast_1d(np.asarray(price, dtype=float))
    S = np.atleast_1d(np.asarray(S, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    price, S, K, T = np.broadcast_arrays(price, S, K, T)
    shape = price.shape
    disc = np.exp(-r * np.maximum(T, _EPS))
    if option == "call":
        lo = np.maximum(S - K * disc, 0.0)
        hi = S
    else:
        lo = np.maximum(K * disc - S, 0.0)
        hi = K * disc
    valid = (price > lo + 1e-10) & (price < hi - 1e-10) & (T > 1e-8)
    sigma = np.full(shape, np.nan)
    sigma[valid] = 0.2
    for _ in range(max_iter):
        if not np.any(valid):
            break
        p = bs_price(S[valid], K[valid], T[valid], r, sigma[valid], option)
        v = bs_vega(S[valid], K[valid], T[valid], r, sigma[valid])
        diff = p - price[valid]
        step = diff / np.maximum(v, 1e-12)
        sigma[valid] = np.clip(sigma[valid] - step, 1e-4, 5.0)
        done = np.abs(diff) < tol
        still = valid.copy()
        still[valid] = ~done
        valid = still
    return sigma.reshape(shape) if shape != () else float(sigma)
