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
