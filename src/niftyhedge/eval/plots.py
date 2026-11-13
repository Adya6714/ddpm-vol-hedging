"""Plotting helpers for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_heatmap(df: pd.DataFrame, path: str | Path, title: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(df.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(list(df.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(list(df.index))
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
