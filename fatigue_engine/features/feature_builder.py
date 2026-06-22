"""Orchestration for fatigue feature engineering."""

from __future__ import annotations

import pandas as pd

from fatigue_engine.config import (
    DATE_COL,
    DEFAULT_INJURY_HISTORY_PATH,
    DEFAULT_POSITION_REPORT_PATH,
    PLAYER_COL,
)
from fatigue_engine.features.intensity_features import build_intensity_features
from fatigue_engine.features.load_features import build_load_features
from fatigue_engine.features.medical_features import build_medical_features
from fatigue_engine.features.position_mapping import apply_position_mapping, write_position_mapping_report
from fatigue_engine.features.recovery_features import build_recovery_features


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, coerce numeric columns, and sort by player chronology."""

    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.dropna(subset=[PLAYER_COL, DATE_COL]).copy()

    non_numeric = {
        "Nom",
        "Player_Name",
        "Team",
        "League",
        "Home_Team",
        "Away_Team",
        "Match_Date",
        "Last_Injury_Category",
        "Next_Injury_Date",
        "Next_Injury_Type",
        "Next_Injury_Category",
        "Medical_Data_Quality_Flag",
    }
    for column in out.columns:
        if column not in non_numeric:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.sort_values([PLAYER_COL, DATE_COL]).reset_index(drop=True)
    out["Position"] = out.get("Position", "Unknown")
    out["Position"] = out["Position"].fillna("Unknown")
    return out


def build_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all raw Fatigue Engine features."""

    out = prepare_base_dataframe(df)
    before_position = out.copy()
    out, ambiguity = apply_position_mapping(out, DEFAULT_INJURY_HISTORY_PATH)
    write_position_mapping_report(
        before_df=before_position,
        after_df=out,
        ambiguity=ambiguity,
        report_path=DEFAULT_POSITION_REPORT_PATH,
    )
    out = build_load_features(out)
    out = build_recovery_features(out)
    out = build_intensity_features(out)
    out = build_medical_features(out)
    return out
