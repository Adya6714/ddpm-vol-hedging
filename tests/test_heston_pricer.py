"""Heston COS pricer sanity check."""

import numpy as np

from niftyhedge.data.iv import bs_price
from niftyhedge.gen.heston import cos_call_price


def test_heston_reduces_toward_bs_when_xi_small():
    S, K, T, r = 100.0, 100.0, 1.0, 0.0
    v0 = theta = 0.04
    px = cos_call_price(S, K, T, r, kappa=2.0, theta=theta, xi=1e-4, rho=0.0, v0=v0, N=256, L=12)
    bs = float(bs_price(S, K, T, r, np.sqrt(v0), "call"))
    # Allow generous tolerance: COS truncation + Heston→BS limit
    assert px > 0
    assert abs(px - bs) / bs < 0.15, f"COS {px} vs BS {bs}"


def test_heston_price_positive_reference_params():
    px = cos_call_price(
        100.0, 100.0, 1.0, 0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, N=256, L=12
    )
    assert px > 0
    bs = float(bs_price(100.0, 100.0, 1.0, 0.0, 0.2, "call"))
    assert 0.3 * bs < px < 3.0 * bs, f"COS {px} vs BS {bs}"
