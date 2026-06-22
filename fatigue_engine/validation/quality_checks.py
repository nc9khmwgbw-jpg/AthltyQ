"""Quality checks for Fatigue Engine datasets."""

from __future__ import annotations

import pandas as pd


def build_quality_summary(df: pd.DataFrame) -> dict[str, float]:
    """Return simple quality metrics for logging and reporting."""

    score_col = "AthltyQ_Fatigue_Index"
    summary: dict[str, float] = {
        "rows": float(len(df)),
        "players": float(df["Nom"].nunique()) if "Nom" in df.columns else 0.0,
    }
    if score_col in df.columns:
        summary["score_missing_rate"] = float(df[score_col].isna().mean())
        summary["score_min"] = float(df[score_col].min())
        summary["score_mean"] = float(df[score_col].mean())
        summary["score_max"] = float(df[score_col].max())
    return summary


def validate_score_bounds(df: pd.DataFrame) -> None:
    """Raise ValueError if score columns leave the expected 0-100 interval."""

    score_columns = [
        "Charge_Score",
        "Recovery_Score",
        "Intensity_Score",
        "Medical_Fragility_Score",
        "AthltyQ_Fatigue_Index",
    ]
    for column in score_columns:
        if column in df.columns and not df[column].dropna().between(0, 100).all():
            raise ValueError(f"{column} contains values outside [0, 100].")
