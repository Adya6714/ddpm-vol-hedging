"""Bootstrap statistics and multiple-testing correction."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_diff(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    stat=None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(stat(a)-stat(b)) with paired resampling."""
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if stat is None:
        stat = np.mean
    rng = np.random.default_rng(seed)
    n = len(loss_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(stat(loss_a[idx]) - stat(loss_b[idx]))
    diffs = np.asarray(diffs)
    est = float(stat(loss_a) - stat(loss_b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return est, float(lo), float(hi)


def block_bootstrap_ci(
    series: np.ndarray,
    stat=None,
    block: int = 30,
    n_boot: int = 5_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    series = np.asarray(series, dtype=float)
    if stat is None:
        stat = np.mean
    rng = np.random.default_rng(seed)
    n = len(series)
    n_blocks = int(np.ceil(n / block))
    samples = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), n_blocks)
        chunks = [series[s : s + block] for s in starts]
        boot = np.concatenate(chunks)[:n]
        samples.append(stat(boot))
    samples = np.asarray(samples)
    est = float(stat(series))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return est, float(lo), float(hi)


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj
