"""Barrier payoff / knock tests."""

import torch

from niftyhedge.hedge.contracts import Contract, barrier_payoff, knock_mask


def test_doc_knockout_zero_payoff():
    S = torch.tensor([[100.0, 95.0, 89.0, 110.0]])  # hits 89 < 90
    contract = Contract(kind="DOC", strike_mult=1.0, barrier_mult=0.90)
    payoff = barrier_payoff(S, contract)
    assert float(payoff[0]) == 0.0


def test_doc_alive_positive():
    S = torch.tensor([[100.0, 101.0, 102.0, 105.0]])
    contract = Contract(kind="DOC", strike_mult=1.0, barrier_mult=0.90)
    payoff = barrier_payoff(S, contract)
    assert float(payoff[0]) == 5.0


def test_knock_mask_sticky():
    S = torch.tensor([[100.0, 89.0, 95.0]])
    contract = Contract(kind="DOC", barrier_mult=0.90)
    m = knock_mask(S, contract)
    assert bool(m[0, 0]) is False
    assert bool(m[0, 1]) is True
    assert bool(m[0, 2]) is True


def _mc_barrier(kind, S0, K, B, T, r, sig, n=400_000, steps=2000, seed=0):
    """Fine-grid MC (close to continuous monitoring) for the closed form."""
    g = torch.Generator().manual_seed(seed)
    dt = T / steps
    logS = torch.full((n,), float(torch.log(torch.tensor(S0))), dtype=torch.float64)
    alive = torch.ones(n, dtype=torch.bool)
    for _ in range(steps):
        logS += (r - 0.5 * sig**2) * dt + sig * dt**0.5 * torch.randn(n, generator=g, dtype=torch.float64)
        S = torch.exp(logS)
        alive &= (S > B) if kind == "DOC" else (S < B)
    pay = torch.clamp(torch.exp(logS) - K, min=0) * alive
    return float(torch.exp(torch.tensor(-r * T)) * pay.mean())


def test_closed_form_doc_and_uoc_match_mc():
    from niftyhedge.hedge.env import bs_barrier_price_torch

    S0, K, T, r, sig = 100.0, 100.0, 30 / 365, 0.065, 0.20
    t = lambda x: torch.tensor([x], dtype=torch.float64)
    for kind, B in [("DOC", 95.0), ("UOC", 105.0)]:
        cf = float(bs_barrier_price_torch(t(S0), t(K), t(B), t(T), r, t(sig), kind=kind))
        mc = _mc_barrier(kind, S0, K, B, T, r, sig, n=200_000, steps=1000)
        assert abs(cf - mc) < 0.03 * max(mc, 0.1) + 0.02, (kind, cf, mc)
    # DOC with far barrier converges to vanilla
    van = float(bs_barrier_price_torch(t(S0), t(K), t(1.0), t(T), r, t(sig), kind="VANILLA_CALL"))
    doc = float(bs_barrier_price_torch(t(S0), t(K), t(1.0), t(T), r, t(sig), kind="DOC"))
    assert abs(van - doc) < 1e-6
