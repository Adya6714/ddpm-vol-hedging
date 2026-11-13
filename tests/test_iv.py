"""IV / BS pricing tests."""

import numpy as np

from niftyhedge.data.iv import bs_price, bs_vega, implied_vol


def test_bs_call_put_parity():
    S, K, T, r, sig = 100.0, 100.0, 1.0, 0.05, 0.2
    c = bs_price(S, K, T, r, sig, "call")
    p = bs_price(S, K, T, r, sig, "put")
    assert abs(c - p - (S - K * np.exp(-r * T))) < 1e-8


def test_implied_vol_roundtrip():
    S, K, T, r, sig = 100.0, 100.0, 0.5, 0.01, 0.25
    px = bs_price(S, K, T, r, sig, "call")
    iv = implied_vol(np.array([px]), np.array([S]), np.array([K]), np.array([T]), r, "call")
    assert abs(float(np.asarray(iv).ravel()[0]) - sig) < 1e-5


def test_vega_positive():
    assert float(bs_vega(100, 100, 1.0, 0.0, 0.2)) > 0
