"""Intensity feature engineering for player fatigue."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fatigue_engine.config import PLAYER_COL


def build_intensity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling intensity features using historical match data only."""

    out = df.copy()
    grouped = out.groupby(PLAYER_COL, group_keys=False)

    out["High_Intensity_Load_Raw"] = out["sprints"] * out["kpi_explosivity"]
    out["Sprints_Per_90_Raw"] = np.where(
        out["Minutes_Played"] > 0,
        out["sprints"] / out["Minutes_Played"] * 90.0,
        np.nan,
    )
    out["Distance_Per_Min_Raw"] = np.where(
        out["Minutes_Played"] > 0,
        out["distanceRun"] / out["Minutes_Played"],
        np.nan,
    )
    out["Defensive_Actions_Raw"] = (
        out["Tackles"] + out["Interceptions"] + out["Clearances"] + out["Ball_Recovery"]
    )

    out["High_Intensity_Load_5"] = grouped["High_Intensity_Load_Raw"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["Sprints_Rolling_5"] = grouped["sprints"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["Sprints_Per_90_Rolling_5"] = grouped["Sprints_Per_90_Raw"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["Explosivity_Rolling_5"] = grouped["kpi_explosivity"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["Distance_Per_Min_Rolling_5"] = grouped["Distance_Per_Min_Raw"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["Defensive_Actions_Rolling_5"] = grouped["Defensive_Actions_Raw"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    return out
