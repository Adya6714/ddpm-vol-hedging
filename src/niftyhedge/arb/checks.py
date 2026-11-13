"""Static arbitrage checks: calendar and Durrleman butterfly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from niftyhedge.data.surface import MONEYNESS, TENORS_DAYS


@dataclass
class ArbReport:
    calendar_violation_frac: float
    butterfly_violation_frac: float
    calendar_penalty: float
    butterfly_penalty: float
    n_surfaces: int


def calendar_violations(surfaces: np.ndarray, tenors: np.ndarray = TENORS_DAYS) -> tuple[np.ndarray, float]:
    """Total variance must be non-decreasing in maturity at fixed moneyness.

    Returns per-surface violation fraction and mean penalty magnitude.
    """
    surfaces = np.asarray(surfaces, dtype=float)
    if surfaces.ndim == 2:
        surfaces = surfaces[None]
    Tdays = np.asarray(tenors, dtype=float) / 365.0
    w = surfaces**2 * Tdays[None, None, :]
    dw = np.diff(w, axis=2)
    viol = dw < -1e-8
    per = viol.any(axis=(1, 2)).astype(float)
    penalty = float(np.mean(np.maximum(-dw, 0.0)))
    return per, penalty


def durrleman_density(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """g(k) = (1 - k*w'/(2w))^2 - (w'/2)^2*(1/w + 1/4) + w''/2  (Gatheral)."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    # finite differences
    dk = np.gradient(k)
    wp = np.gradient(w, k)
    wpp = np.gradient(wp, k)
    term1 = (1.0 - k * wp / (2.0 * np.maximum(w, 1e-12))) ** 2
    term2 = 0.25 * wp**2 * (1.0 / np.maximum(w, 1e-12) + 0.25)
    g = term1 - term2 + 0.5 * wpp
    return g


def butterfly_violations(
    surfaces: np.ndarray,
    moneyness: np.ndarray = MONEYNESS,
    tenors: np.ndarray = TENORS_DAYS,
) -> tuple[np.ndarray, float]:
    """Durrleman g(k) >= 0 on each maturity slice."""
    surfaces = np.asarray(surfaces, dtype=float)
    if surfaces.ndim == 2:
        surfaces = surfaces[None]
    k = np.log(np.asarray(moneyness, dtype=float))
    T = np.asarray(tenors, dtype=float) / 365.0
    n = surfaces.shape[0]
    viol_any = np.zeros(n, dtype=bool)
    penalties = []
    for i in range(n):
        bad = False
        pen = 0.0
        for j in range(len(T)):
            w = surfaces[i, :, j] ** 2 * T[j]
            g = durrleman_density(k, w)
            neg = np.minimum(g, 0.0)
            if np.any(neg < -1e-6):
                bad = True
            pen += float(np.mean(np.maximum(-g, 0.0)))
        viol_any[i] = bad
        penalties.append(pen / len(T))
    return viol_any.astype(float), float(np.mean(penalties))


def report_arbitrage(surfaces: np.ndarray) -> ArbReport:
    cal_f, cal_p = calendar_violations(surfaces)
    but_f, but_p = butterfly_violations(surfaces)
    return ArbReport(
        calendar_violation_frac=float(cal_f.mean()),
        butterfly_violation_frac=float(but_f.mean()),
        calendar_penalty=cal_p,
        butterfly_penalty=but_p,
        n_surfaces=int(np.asarray(surfaces).reshape(-1, 9, 6).shape[0])
        if np.asarray(surfaces).ndim >= 2
        else 1,
    )


def invert_calendar_slice(surface: np.ndarray) -> np.ndarray:
    """Deliberately break calendar: make short-dated TV exceed long-dated TV."""
    s = np.array(surface, copy=True, dtype=float)
    # Set 7d IV extremely high so w_7d > w_180d
    s[:, 0] = 1.5
    s[:, -1] = 0.05
    return s
