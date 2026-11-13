"""Risk measure tests."""

import torch

from niftyhedge.hedge.risk import cvar, ru_objective, var


def test_cvar_ge_var():
    loss = torch.arange(100.0)
    assert float(cvar(loss, 0.95)) >= float(var(loss, 0.95)) - 1e-6


def test_ru_matches_cvar_at_optimum():
    loss = torch.randn(5000)
    # optimize w
    w = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        ru_objective(loss, w, 0.95).backward()
        opt.step()
    assert abs(float(ru_objective(loss, w.detach(), 0.95)) - float(cvar(loss, 0.95))) < 0.05
