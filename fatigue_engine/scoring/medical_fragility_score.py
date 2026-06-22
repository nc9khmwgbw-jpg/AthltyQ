"""Medical fragility score calculation."""

from __future__ import annotations

import pandas as pd


def add_medical_fragility_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add Medical_Fragility_Score with neutral handling for missing histories."""

    out = df.copy()
    out["Medical_Fragility_Score"] = (
        0.25 * out["N_Injury_Count_12M_Before_T"]
        + 0.20 * out["N_INV_Days_Since_Last_Injury"]
        + 0.20 * out["N_Total_Injury_Days_12M"]
        + 0.15 * out["N_Muscle_Injury_Count_12M"]
        + 0.10 * out["Recurring_Same_Category_12M"]
        + 0.10 * out["N_Last_Injury_Duration_Days"]
    ).clip(0.0, 100.0)

    if "Injury_History_Available" in out.columns:
        missing_history = out["Injury_History_Available"].fillna(0) == 0
        out.loc[missing_history, "Medical_Fragility_Score"] = 35.0
    return out
