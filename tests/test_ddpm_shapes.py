"""DDPM network shape tests."""

import torch

from niftyhedge.gen.ddpm import Denoiser, cosine_alpha_bars


def test_denoiser_shapes():
    net = Denoiser(x_dim=55, cond_dim=112, hidden=64)
    x = torch.randn(8, 55)
    t = torch.rand(8, 1)
    c = torch.randn(8, 112)
    y = net(x, t, c)
    assert y.shape == (8, 55)


def test_cosine_schedule_monotonic():
    ab = cosine_alpha_bars(50, "cpu")
    assert ab[0] == 1.0 or abs(float(ab[0]) - 1.0) < 1e-6
    assert torch.all(ab[1:] <= ab[:-1] + 1e-6)
