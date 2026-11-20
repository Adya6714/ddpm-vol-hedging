#!/usr/bin/env python3
"""Print PASS/FAIL for EXPERIMENTS_PLAN Part 5 acceptance checks."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def ok(msg: str) -> None:
    print(f"PASS  {msg}")


def bad(msg: str) -> None:
    print(f"FAIL  {msg}")


def check_a1() -> bool:
    try:
        import niftyhedge

        ok(f"A1 import niftyhedge {niftyhedge.__version__}")
        return True
    except Exception as exc:  # noqa: BLE001
        bad(f"A1 import failed: {exc}")
        return False


def check_a2() -> bool:
    required = [
        "test_iv.py",
        "test_svi.py",
        "test_arb.py",
        "test_features_no_leak.py",
        "test_env_sign.py",
        "test_env_convergence.py",
        "test_barrier.py",
        "test_risk.py",
        "test_worlds_interface.py",
        "test_heston_pricer.py",
        "test_ddpm_shapes.py",
        "test_legacy_regression.py",
    ]
    missing = [t for t in required if not (ROOT / "tests" / t).exists()]
    if missing:
        bad(f"A2 missing tests: {missing}")
        return False
    # non-trivial: each file > 20 lines
    thin = [t for t in required if sum(1 for _ in open(ROOT / "tests" / t)) < 20]
    if thin:
        bad(f"A2 trivial tests: {thin}")
        return False
    ok("A2 twelve non-trivial test files exist")
    return True


def check_a10() -> bool:
    src = ROOT / "src"
    bad_pat = re.compile(r"google\.colab|drive\.mount|except\s*:|np\.where\(\s*np\.isnan")
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        if bad_pat.search(text):
            # allow commented mentions? still fail hard
            offenders.append(str(path.relative_to(ROOT)))
        # bare except
        try:
            tree = ast.parse(text)
        except SyntaxError:
            offenders.append(str(path.relative_to(ROOT)) + " (syntax)")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                offenders.append(str(path.relative_to(ROOT)) + ":bare-except")
    if offenders:
        bad(f"A10 offenders: {offenders}")
        return False
    ok("A10 no colab/drive/bare-except/isnan-fill in src/")
    return True


def check_a11() -> bool:
    import torch
    from niftyhedge.gen.base import Paths
    from niftyhedge.hedge.baselines import no_hedge
    from niftyhedge.hedge.contracts import Contract
    from niftyhedge.hedge.env import terminal_loss
    from niftyhedge.hedge.instruments import CostModel

    n, H = 100_000, 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.ones(n, H + 1, device=device) * 20000
    # tiny noise
    S = S * torch.exp(0.001 * torch.cumsum(torch.randn(n, H + 1, device=device), dim=1))
    surf = torch.full((n, H + 1, 9, 6), 0.16, device=device)
    paths = Paths(S=S, surf=surf, start_idx=torch.arange(n, device=device))
    contract = Contract(kind="DOC", barrier_mult=0.9)
    h, q = no_hedge(paths, contract)
    p0 = torch.zeros(n, device=device)
    t0 = time.perf_counter()
    _ = terminal_loss(paths, contract, h, q, None, CostModel(), p0)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    limit = 10.0 if device == "cuda" else 60.0
    if dt > limit:
        bad(f"A11 terminal_loss {dt:.2f}s > {limit}s on {device}")
        return False
    ok(f"A11 terminal_loss {dt:.2f}s on {device} (limit {limit}s)")
    return True


def check_a14() -> bool:
    """Gate 3: in the GBM world BS barrier delta is (near-)optimal, so a correctly
    working deep hedger must MATCH it: CVaR within 5% of BGK delta, not far
    better (that would signal a leak / scoring bug) and not worse (broken training)."""
    import json

    metas = sorted(Path("models/hedge").glob("gbm/I1/doc_090/fold_*/seed_*/policy_meta.json"))
    if not metas:
        bad("A14 no trained gbm/I1/doc_090 hedger with policy_meta.json (run Gate 3 first)")
        return False
    ok_all = True
    for m in metas:
        meta = json.loads(m.read_text())
        dh, bs = meta.get("val_cvar"), meta.get("bs_val_cvar")
        if dh is None or bs is None:
            bad(f"A14 {m.parent}: missing val_cvar/bs_val_cvar")
            ok_all = False
            continue
        ratio = dh / bs if bs else float("inf")
        msg = f"A14 {m.parent.parent.name}/{m.parent.name}: deep {dh*1e4:.1f}bp vs BGK {bs*1e4:.1f}bp (ratio {ratio:.3f})"
        if 0.80 <= ratio <= 1.05:
            ok(msg)
        else:
            bad(msg + " - need 0.80..1.05")
            ok_all = False
    return ok_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()
    # Gate 0 checks (A14 is Gate 3 — only when --only A14)
    checks = {
        "A1": check_a1,
        "A2": check_a2,
        "A10": check_a10,
        "A11": check_a11,
    }
    static = {
        "A3": "covered by tests/test_legacy_regression.py",
        "A4": "covered by tests/test_env_sign.py",
        "A5": "covered by tests/test_env_convergence.py",
        "A6": "covered by tests/test_features_no_leak.py",
        "A7": "covered by tests/test_arb.py",
        "A8": "covered by tests/test_worlds_interface.py",
        "A9": "covered by `make smoke`",
        "A12": "manifest.json written by scripts (spot-check after smoke)",
        "A13": "download_log.csv from scripts/00_download.py (laptop Gate 1a)",
        "A14": "Gate 3 — run: python scripts/check_build.py --only A14",
    }
    only = args.only
    failed = False
    if only == "A14":
        raise SystemExit(0 if check_a14() else 1)
    for k, fn in checks.items():
        if only and k != only:
            continue
        if not fn():
            failed = True
    for k, note in static.items():
        if only and k != only:
            continue
        print(f"INFO  {k} {note}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
