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


def test_arb_free_surface_positive():
    surf = make_arb_free_surface(MONEYNESS, TENORS_DAYS / 365.0)
    assert surf.shape == (9, 6)
    assert np.all(surf > 0)
