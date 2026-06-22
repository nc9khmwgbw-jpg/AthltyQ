"""Leakage validation for Fatigue Engine outputs."""

from __future__ import annotations

import pandas as pd

from fatigue_engine.config import LEAKAGE_COLUMNS


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove known future-target columns from the exported dataset."""

    return df.drop(columns=sorted(LEAKAGE_COLUMNS & set(df.columns)), errors="ignore")


def validate_no_leakage_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if known leakage columns are present."""

    present = sorted(LEAKAGE_COLUMNS & set(df.columns))
    if present:
        raise ValueError(f"Leakage columns found in fatigue dataset: {present}")
