"""Intensity score calculation."""

from __future__ import annotations

import pandas as pd


def add_intensity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add Intensity_Score using normalized neuromuscular load features."""

    out = df.copy()
    out["Intensity_Score"] = (
        0.30 * out["N_High_Intensity_Load_5"]
        + 0.25 * out["N_Sprints_Per_90_Rolling_5"]
        + 0.20 * out["N_Explosivity_Rolling_5"]
        + 0.15 * out["N_Distance_Per_Min_Rolling_5"]
        + 0.10 * out["N_Defensive_Actions_Rolling_5"]
    ).clip(0.0, 100.0)
    return out
