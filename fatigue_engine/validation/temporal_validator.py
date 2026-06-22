"""Temporal validation helpers."""

from __future__ import annotations

import pandas as pd

from fatigue_engine.config import DATE_COL, PLAYER_COL


def validate_temporal_order(df: pd.DataFrame) -> None:
    """Ensure player match rows are chronologically ordered."""

    ordered = df.sort_values([PLAYER_COL, DATE_COL]).index
    if not ordered.equals(df.index):
        raise ValueError("Fatigue dataset must be sorted by player and Match_Date.")


def validate_rolling_features_have_history(df: pd.DataFrame) -> None:
    """Ensure rows with scores generally have historical context."""

    if "Minutes_Rolling_5" not in df.columns:
        raise ValueError("Minutes_Rolling_5 is missing; load features were not built.")
