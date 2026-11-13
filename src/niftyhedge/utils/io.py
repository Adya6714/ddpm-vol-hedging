"""Run directories, manifests, NaN guards."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_manifest(
    run_dir: str | Path,
    *,
    config: dict[str, Any],
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_dir = ensure_dir(run_dir)
    manifest = {
        "config": config,
        "config_hash": config_hash(config),
        "seed": seed,
        "git_hash": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def manifest_matches(run_dir: str | Path, config: dict[str, Any]) -> bool:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return False
    existing = json.loads(path.read_text())
    return existing.get("config_hash") == config_hash(config)


def assert_finite(x: np.ndarray | torch.Tensor, name: str = "tensor") -> None:
    if isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            raise ValueError(f"NaN/Inf in {name}")
    else:
        if not np.isfinite(x).all():
            raise ValueError(f"NaN/Inf in {name}")
