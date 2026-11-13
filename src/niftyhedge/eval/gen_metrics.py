"""Generator quality metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def crps_gaussian(obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """CRPS for Gaussian predictive distribution (closed form)."""
    obs, mu, sigma = np.asarray(obs), np.asarray(mu), np.asarray(sigma)
    sigma = np.maximum(sigma, 1e-8)
    z = (obs - mu) / sigma
    from scipy.stats import norm

    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def energy_score(samples: np.ndarray, obs: np.ndarray) -> float:
    """samples [m, d], obs [d]."""
    samples = np.asarray(samples)
    obs = np.asarray(obs)
    m = len(samples)
    t1 = np.mean(np.linalg.norm(samples - obs[None, :], axis=1))
    if m < 2:
        return float(t1)
    # pairwise
    diff = samples[:, None, :] - samples[None, :, :]
    t2 = np.mean(np.linalg.norm(diff, axis=-1))
    return float(t1 - 0.5 * t2)


def marginal_distances(real: np.ndarray, gen: np.ndarray) -> dict:
    real, gen = np.asarray(real).ravel(), np.asarray(gen).ravel()
    return {
        "ks": float(ks_2samp(real, gen).statistic),
        "w1": float(wasserstein_distance(real, gen)),
    }


def atm_stats(surfaces: np.ndarray) -> dict:
    atm = np.asarray(surfaces)[:, 4, 2]
    return {"mean": float(atm.mean()), "std": float(atm.std())}
