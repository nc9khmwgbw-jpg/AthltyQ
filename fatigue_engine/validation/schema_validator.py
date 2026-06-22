"""Schema validation for fatigue input data."""

from __future__ import annotations

import pandas as pd

from fatigue_engine.config import REQUIRED_COLUMNS


def validate_input_schema(df: pd.DataFrame) -> None:
    """Raise ValueError when required input columns are missing."""

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required fatigue input columns: {missing}")
