"""Normalization pipeline for fatigue features."""

from __future__ import annotations

import pandas as pd

from fatigue_engine.config import (
    GROUP_COLS,
    INVERSE_NORMALIZATION_FEATURES,
    POSITIVE_NORMALIZATION_FEATURES,
)
from fatigue_engine.normalization.robust_scaler import robust_score


def normalize_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized 0-100 feature columns used by scoring."""

    out = df.copy()
    for column in sorted(POSITIVE_NORMALIZATION_FEATURES):
        if column in out.columns:
            out[f"N_{column}"] = robust_score(out, column, GROUP_COLS)

    for column in sorted(INVERSE_NORMALIZATION_FEATURES):
        if column in out.columns:
            out[f"N_INV_{column}"] = robust_score(out, column, GROUP_COLS, inverse=True)

    for flag in ("Short_Rest_Flag", "Congested_Period_Flag", "Recurring_Same_Category_12M"):
        if flag in out.columns:
            out[flag] = pd.to_numeric(out[flag], errors="coerce").fillna(0.0).clip(0.0, 100.0)

    return out
