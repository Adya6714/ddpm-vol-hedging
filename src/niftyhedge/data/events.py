"""Pre-announced event calendar only (no hindsight crash days)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

# Illustrative RBI MPC / Budget / election counting days used for conditioning.
# Extend from official calendars when rebuilding the dataset.
EVENT_DATES: set[date] = {
    date(2018, 2, 1),
    date(2019, 2, 1),
    date(2019, 5, 23),
    date(2020, 2, 1),
    date(2021, 2, 1),
    date(2022, 2, 1),
    date(2023, 2, 1),
    date(2024, 2, 1),
    date(2024, 6, 4),
    date(2025, 2, 1),
    date(2026, 2, 1),
}


def to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if hasattr(d, "item"):
        item = d.item()
        if isinstance(item, datetime):
            return item.date()
        if isinstance(item, date):
            return item
    return pd.Timestamp(d).date()


def is_event(d) -> bool:
    return to_date(d) in EVENT_DATES


def event_flag_series(dates) -> list[int]:
    return [1 if is_event(d) else 0 for d in dates]
