"""Leak-free conditioning tests."""

import numpy as np

from niftyhedge.data.features import build_conditioning
from niftyhedge.data.surface import make_sample_surfaces


def test_perturb_day_t_leaves_row_t_unchanged():
    grid = make_sample_surfaces(40, seed=1)
    spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.01 * np.random.randn(39)])
    vix = 14 + np.random.randn(40)
    C0 = build_conditioning(grid.surfaces, grid.dates, spot, vix, lag=1)

    t = 20
    surfaces2 = grid.surfaces.copy()
    surfaces2[t] *= 1.5
    spot2 = spot.copy()
    spot2[t] *= 1.1
    vix2 = vix.copy()
    vix2[t] = 99.0
    C1 = build_conditioning(surfaces2, grid.dates, spot2, vix2, lag=1)

    # event column may differ only if we changed events; we didn't
    # conditioning row t must ignore day-t market data
    # exclude event flag column (last) — same
    assert np.allclose(C0[t, :-1], C1[t, :-1], atol=1e-10)


def test_lag_must_be_at_least_one():
    grid = make_sample_surfaces(10, seed=0)
    spot = np.full(10, 20000.0)
    vix = np.full(10, 14.0)
    try:
        build_conditioning(grid.surfaces, grid.dates, spot, vix, lag=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
