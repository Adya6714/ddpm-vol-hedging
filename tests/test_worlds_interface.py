"""All five worlds implement World contract."""

import numpy as np
import torch

from niftyhedge.data.features import build_conditioning
from niftyhedge.data.splits import FoldData
from niftyhedge.data.surface import make_sample_surfaces
from niftyhedge.gen.base import start_states_from_fold
from niftyhedge.gen.bootstrap import BootstrapWorld
from niftyhedge.gen.ddpm import DDPMWorld
from niftyhedge.gen.gbm import GBMWorld
from niftyhedge.gen.heston import HestonWorld
from niftyhedge.gen.pcavar import PCAVARWorld


def _fold():
    grid = make_sample_surfaces(80, seed=0)
    spot = 20000 * np.cumprod(np.r_[1.0, 1 + 0.005 * np.random.RandomState(0).randn(79)])
    vix = np.full(80, 14.0)
    C = build_conditioning(grid.surfaces, grid.dates, spot, vix, lag=1)
    n = 50
    return FoldData(
        fold_id=0,
        X_train=grid.surfaces[:n],
        X_val=grid.surfaces[n : n + 15],
        X_test=grid.surfaces[n + 15 :],
        C_train=C[:n],
        C_val=C[n : n + 15],
        C_test=C[n + 15 :],
        dates_train=grid.dates[:n],
        dates_val=grid.dates[n : n + 15],
        dates_test=grid.dates[n + 15 :],
        spot_train=spot[:n],
        spot_val=spot[n : n + 15],
        spot_test=spot[n + 15 :],
        log_mean=np.zeros((9, 6)),
        log_std=np.ones((9, 6)),
    )


def _check_world(world, fold, n_paths=8, horizon=5):
    world.fit(fold, seed=0)
    starts = start_states_from_fold(fold, "train")
    paths = world.sample_paths(starts, n_paths=n_paths, horizon=horizon, seed=1)
    paths.validate()
    assert paths.S.shape == (n_paths, horizon + 1)
    assert paths.surf.shape == (n_paths, horizon + 1, 9, 6)
    assert torch.all(paths.surf > 0)
    # S[:,0] equals a real start spot
    assert torch.all(torch.isin(paths.S[:, 0].cpu(), starts.spot.cpu()))


def test_gbm_world():
    _check_world(GBMWorld(), _fold())


def test_heston_world():
    _check_world(HestonWorld(), _fold())


def test_pcavar_world():
    _check_world(PCAVARWorld(n_factors=3), _fold())


def test_bootstrap_world():
    _check_world(BootstrapWorld(block=3), _fold())


def test_ddpm_world():
    _check_world(DDPMWorld(timesteps=5, hidden=32, epochs=1), _fold(), n_paths=4, horizon=3)
