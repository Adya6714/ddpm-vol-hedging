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


def paired_block_bootstrap(
    loss_a: np.ndarray,
    loss_b: np.ndarray | None = None,
    stat=None,
    block: int = 30,
    n_boot: int = 2_000,
    seed: int = 0,
) -> dict:
    """Moving-block bootstrap for stat(a) and stat(a) - stat(b) with the SAME blocks.

    Use for overlapping real-market windows (consecutive start dates share up
    to `block`-1 days, so an i.i.d. bootstrap would understate the CI).
    Returns est/lo/hi for a, and diff/diff_lo/diff_hi/p_two_sided when b given.
    """
    a = np.asarray(loss_a, dtype=float)
    b = None if loss_b is None else np.asarray(loss_b, dtype=float)
    stat = stat or np.mean
    rng = np.random.default_rng(seed)
    n = len(a)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    sa, sd = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        sa[i] = stat(a[idx])
        if b is not None:
            sd[i] = stat(a[idx]) - stat(b[idx])
    out = {"est": float(stat(a)), "lo": float(np.quantile(sa, 0.025)), "hi": float(np.quantile(sa, 0.975))}
    if b is not None:
        d = float(stat(a) - stat(b))
        out.update(diff=d, diff_lo=float(np.quantile(sd, 0.025)), diff_hi=float(np.quantile(sd, 0.975)),
                   p_two_sided=float(min(1.0, 2 * min((sd <= 0).mean(), (sd >= 0).mean()))))
    return out


def cvar_np(x: np.ndarray, alpha: float = 0.95) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(int(np.ceil((1 - alpha) * len(x))), 1)
    return float(x[:k].mean())
