"""SVI fit tests."""

import numpy as np

from niftyhedge.data.svi import fit_svi_slice, make_arb_free_surface, svi_total_variance, SVIParams
from niftyhedge.data.surface import MONEYNESS, TENORS_DAYS


def test_svi_fit_recovers_params():
    k = np.linspace(-0.3, 0.3, 11)
    true = SVIParams(a=0.04, b=0.1, rho=-0.4, m=0.0, sigma=0.15)
    w = svi_total_variance(k, true)
    fit, rmse = fit_svi_slice(k, w)
    assert rmse < 1e-3
    assert abs(fit.a - true.a) < 0.02


def test_svi_fit_short_dated_slice():
    """9-day slice: w ~ 4e-4. Regression for the unscaled-optimiser bug."""
    T = 9 / 365
    k = np.linspace(-0.08, 0.04, 15)
    true = SVIParams(a=0.13**2 * T * 0.6, b=0.13**2 * T * 2.0, rho=-0.5, m=0.005, sigma=0.03)
    w = svi_total_variance(k, true)
    fit, _ = fit_svi_slice(k, w)
    iv_fit = np.sqrt(svi_total_variance(k, fit) / T)
    iv_true = np.sqrt(w / T)
    assert np.max(np.abs(iv_fit - iv_true)) < 0.005


def test_arb_free_surface_positive():
    surf = make_arb_free_surface(MONEYNESS, TENORS_DAYS / 365.0)
    assert surf.shape == (9, 6)
    assert np.all(surf > 0)
