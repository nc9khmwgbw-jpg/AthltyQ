"""Recovery score calculation."""

from __future__ import annotations

import pandas as pd


def add_recovery_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add Recovery_Score using normalized rest and congestion features."""

    out = df.copy()
    out["Recovery_Score"] = (
        0.35 * out["N_INV_Days_Rest_Before_T1"]
        + 0.20 * out["N_INV_Days_Rest_Before_T"]
        + 0.20 * out["N_Matches_Last_14D"]
        + 0.15 * out["N_Matches_Last_7D"]
        + 0.10 * out["Short_Rest_Flag"]
    ).clip(0.0, 100.0)
    return out
