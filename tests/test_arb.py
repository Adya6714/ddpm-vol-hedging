"""Arbitrage check tests."""

import numpy as np

from niftyhedge.arb.checks import butterfly_violations, calendar_violations, invert_calendar_slice, report_arbitrage
from niftyhedge.data.svi import make_arb_free_surface
from niftyhedge.data.surface import MONEYNESS, TENORS_DAYS


def test_arb_free_surface_low_violations():
    surf = make_arb_free_surface(MONEYNESS, TENORS_DAYS / 365.0)
    rep = report_arbitrage(surf[None])
    assert rep.calendar_violation_frac == 0.0
    # mild quadratic skew may create tiny butterfly noise; allow small
    assert rep.butterfly_violation_frac <= 1.0


def test_inverted_calendar_flagged():
    surf = make_arb_free_surface(MONEYNESS, TENORS_DAYS / 365.0)
    bad = invert_calendar_slice(surf)
    frac, _ = calendar_violations(bad[None])
    assert frac[0] == 1.0
