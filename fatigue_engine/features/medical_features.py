"""Medical fragility feature preparation."""

from __future__ import annotations

import pandas as pd

from fatigue_engine.config import MEDICAL_COLUMNS


def build_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure medical features exist and are numerically usable."""

    out = df.copy()
    for column in MEDICAL_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["Medical_Data_Quality_Flag"] = "available"
    if "Injury_History_Available" in out.columns:
        missing_history = out["Injury_History_Available"].fillna(0) == 0
        out.loc[missing_history, "Medical_Data_Quality_Flag"] = "missing_history"
    return out
