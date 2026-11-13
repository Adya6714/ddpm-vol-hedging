"""Reproduce Part 1 audit ordering on saved checkpoints (CPU)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "legacy" / "audit_env.py"
AGENT = ROOT / "models" / "rl" / "best_agent_v2.pt"
SCEN = ROOT / "data" / "scenarios" / "scenarios_normal.npy"


@pytest.mark.skipif(not AGENT.exists() or not SCEN.exists(), reason="legacy artefacts missing")
def test_legacy_audit_ordering_normal():
    # Fewer episodes for CI speed; ordering is stable
    n = 50
    proc = subprocess.run(
        [sys.executable, str(AUDIT), str(ROOT), str(n)],
        capture_output=True,
        text=True,
        check=True,
    )
    text = proc.stdout
    assert "== normal" in text
    metrics = {}
    for line in text.splitlines():
        if "metric=" not in line:
            continue
        if "BS +delta" in line:
            metrics["bs_plus"] = float(re.search(r"metric=([+-]?[0-9.]+)", line).group(1))
        elif "No hedge" in line:
            metrics["no"] = float(re.search(r"metric=([+-]?[0-9.]+)", line).group(1))
        elif "DDPM-RL" in line:
            metrics["ddpm"] = float(re.search(r"metric=([+-]?[0-9.]+)", line).group(1))
        elif "BS -delta" in line:
            metrics["bs_minus"] = float(re.search(r"metric=([+-]?[0-9.]+)", line).group(1))
    assert set(metrics) == {"bs_plus", "no", "ddpm", "bs_minus"}
    # less negative is better: bs_plus < no < ddpm < bs_minus
    assert metrics["bs_plus"] < metrics["no"] < metrics["ddpm"] < metrics["bs_minus"]
