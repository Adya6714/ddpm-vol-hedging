"""Correct hedge sign beats no-hedge; wrong sign is worse."""

import torch

from niftyhedge.data.splits import FoldData
from niftyhedge.gen.base import start_states_from_fold
from niftyhedge.gen.gbm import GBMWorld
from niftyhedge.hedge.baselines import bs_barrier_delta_bgk, no_hedge
from niftyhedge.hedge.contracts import Contract
from niftyhedge.hedge.env import terminal_loss
from niftyhedge.hedge.instruments import CostModel
from niftyhedge.hedge.risk import cvar
from niftyhedge.data.surface import make_sample_surfaces
from niftyhedge.data.features import build_conditioning
import numpy as np


def _tiny_fold():
    grid = make_sample_surfaces(60, seed=0)
    spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.005 * np.random.RandomState(0).randn(59)])
    vix = np.full(60, 14.0)
    C = build_conditioning(grid.surfaces, grid.dates, spot, vix, lag=1)
    n = 40
    return FoldData(
        fold_id=0,
        X_train=grid.surfaces[:n],
        X_val=grid.surfaces[n : n + 10],
        X_test=grid.surfaces[n + 10 :],
        C_train=C[:n],
        C_val=C[n : n + 10],
        C_test=C[n + 10 :],
        dates_train=grid.dates[:n],
        dates_val=grid.dates[n : n + 10],
        dates_test=grid.dates[n + 10 :],
        spot_train=spot[:n],
        spot_val=spot[n : n + 10],
        spot_test=spot[n + 10 :],
        log_mean=np.zeros((9, 6)),
        log_std=np.ones((9, 6)),
    )


def test_correct_delta_beats_no_hedge_and_wrong_sign():
    fold = _tiny_fold()
    world = GBMWorld()
    world.fit(fold, 0)
    starts = start_states_from_fold(fold, "train")
    paths = world.sample_paths(starts, n_paths=2000, horizon=30, seed=1)
    contract = Contract(kind="DOC", barrier_mult=0.90)
    cost = CostModel(underlying_bps=0.0, option_frac=0.0)
    p0 = torch.zeros(paths.S.shape[0])

    h0, _ = no_hedge(paths, contract)
    h_ok, _ = bs_barrier_delta_bgk(paths, contract)
    h_bad = -h_ok

    L0 = terminal_loss(paths, contract, h0, None, None, cost, p0)
    Lok = terminal_loss(paths, contract, h_ok, None, None, cost, p0)
    Lbad = terminal_loss(paths, contract, h_bad, None, None, cost, p0)

    c0, cok, cbad = float(cvar(L0)), float(cvar(Lok)), float(cvar(Lbad))
    assert cok < 0.5 * c0, f"correct delta CVaR {cok} should be < 50% of no-hedge {c0}"
    assert cbad > c0, f"wrong-sign {cbad} should be worse than no-hedge {c0}"
