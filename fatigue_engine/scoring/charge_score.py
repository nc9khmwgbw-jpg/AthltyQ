"""Charge score calculation."""

from __future__ import annotations

import pandas as pd


def add_charge_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add Charge_Score using normalized load features."""

    out = df.copy()
    out["Charge_Score"] = (
        0.25 * out["N_Minutes_Rolling_5"]
        + 0.15 * out["N_Minutes_Rolling_10"]
        + 0.20 * out["N_Distance_Rolling_5"]
        + 0.20 * out["N_Full_Matches_Rolling_5"]
        + 0.20 * out["N_Work_Rate_Rolling_5"]
    ).clip(0.0, 100.0)
    return out
