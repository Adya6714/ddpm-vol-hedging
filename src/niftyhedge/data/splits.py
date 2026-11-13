"""Walk-forward folds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from niftyhedge.data.events import to_date


@dataclass
class FoldData:
    fold_id: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    C_train: np.ndarray
    C_val: np.ndarray
    C_test: np.ndarray
    dates_train: np.ndarray
    dates_val: np.ndarray
    dates_test: np.ndarray
    spot_train: np.ndarray
    spot_val: np.ndarray
    spot_test: np.ndarray
    log_mean: np.ndarray
    log_std: np.ndarray


# Test year for fold k (train up to end of prior year)
DEFAULT_FOLD_TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def make_walk_forward_folds(
    surfaces: np.ndarray,
    conditioning: np.ndarray,
    dates,
    spot: np.ndarray,
    out_dir: str | Path,
    test_years: list[int] | None = None,
    val_months: int = 6,
) -> list[FoldData]:
    """Create walk-forward folds; normalisation stats from train only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_years = test_years or DEFAULT_FOLD_TEST_YEARS
    years = np.array([to_date(d).year for d in dates])
    months = np.array([to_date(d).month for d in dates])
    folds: list[FoldData] = []
    manifest = {"folds": []}

    for k, ty in enumerate(test_years):
        train_mask = years < ty
        test_mask = (years == ty) | ((ty == 2025) & (years == 2026))
        if ty == 2025:
            test_mask = (years == 2025) | (years == 2026)
        if not train_mask.any() or not test_mask.any():
            # allow smoke/synthetic data with short span: fall back to index split
            n = len(dates)
            n_train = max(int(0.6 * n), 10)
            n_val = max(int(0.2 * n), 5)
            idx = np.arange(n)
            tr, va, te = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
        else:
            train_idx = np.where(train_mask)[0]
            # val = last val_months of training span
            last_train_year = ty - 1
            val_mask = train_mask & (
                ((years == last_train_year) & (months > 12 - val_months))
                | ((years == last_train_year) & (months >= 7) & (val_months >= 6))
            )
            # simpler: last 6 calendar months before test year
            cutoff_month_start = 7  # Jul–Dec of year ty-1
            val_mask = train_mask & (years == last_train_year) & (months >= cutoff_month_start)
            if not val_mask.any():
                # take last 20% of train_idx
                cut = max(len(train_idx) - max(len(train_idx) // 5, 1), 1)
                tr, va = train_idx[:cut], train_idx[cut:]
            else:
                va = np.where(val_mask)[0]
                tr = np.where(train_mask & ~val_mask)[0]
            te = np.where(test_mask)[0]

        def _norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            logX = np.log(np.clip(X, 1e-6, None))
            mu = logX.mean(axis=0)
            sd = logX.std(axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            return mu, sd

        mu, sd = _norm_stats(surfaces[tr])

        def pack(idx):
            return (
                surfaces[idx],
                conditioning[idx],
                np.asarray(dates)[idx],
                spot[idx],
            )

        Xt, Ct, Dt, St = pack(tr)
        Xv, Cv, Dv, Sv = pack(va)
        Xe, Ce, De, Se = pack(te)

        fold_dir = out_dir / f"fold_{k}"
        fold_dir.mkdir(exist_ok=True)
        np.save(fold_dir / "X_train.npy", Xt)
        np.save(fold_dir / "X_val.npy", Xv)
        np.save(fold_dir / "X_test.npy", Xe)
        np.save(fold_dir / "C_train.npy", Ct)
        np.save(fold_dir / "C_val.npy", Cv)
        np.save(fold_dir / "C_test.npy", Ce)
        np.save(fold_dir / "dates_train.npy", Dt)
        np.save(fold_dir / "dates_val.npy", Dv)
        np.save(fold_dir / "dates_test.npy", De)
        np.save(fold_dir / "spot_train.npy", St)
        np.save(fold_dir / "spot_val.npy", Sv)
        np.save(fold_dir / "spot_test.npy", Se)
        np.save(fold_dir / "log_mean.npy", mu)
        np.save(fold_dir / "log_std.npy", sd)

        fold = FoldData(
            fold_id=k,
            X_train=Xt,
            X_val=Xv,
            X_test=Xe,
            C_train=Ct,
            C_val=Cv,
            C_test=Ce,
            dates_train=Dt,
            dates_val=Dv,
            dates_test=De,
            spot_train=St,
            spot_val=Sv,
            spot_test=Se,
            log_mean=mu,
            log_std=sd,
        )
        folds.append(fold)
        manifest["folds"].append(
            {
                "fold_id": k,
                "test_year": ty,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
            }
        )

    (out_dir / "fold_manifest.json").write_text(json.dumps(manifest, indent=2))
    return folds


def load_fold(fold_dir: str | Path) -> FoldData:
    fold_dir = Path(fold_dir)
    fid = int(fold_dir.name.split("_")[1])
    return FoldData(
        fold_id=fid,
        X_train=np.load(fold_dir / "X_train.npy"),
        X_val=np.load(fold_dir / "X_val.npy"),
        X_test=np.load(fold_dir / "X_test.npy"),
        C_train=np.load(fold_dir / "C_train.npy"),
        C_val=np.load(fold_dir / "C_val.npy"),
        C_test=np.load(fold_dir / "C_test.npy"),
        dates_train=np.load(fold_dir / "dates_train.npy", allow_pickle=True),
        dates_val=np.load(fold_dir / "dates_val.npy", allow_pickle=True),
        dates_test=np.load(fold_dir / "dates_test.npy", allow_pickle=True),
        spot_train=np.load(fold_dir / "spot_train.npy"),
        spot_val=np.load(fold_dir / "spot_val.npy"),
        spot_test=np.load(fold_dir / "spot_test.npy"),
        log_mean=np.load(fold_dir / "log_mean.npy"),
        log_std=np.load(fold_dir / "log_std.npy"),
    )
