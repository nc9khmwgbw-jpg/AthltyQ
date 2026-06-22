"""Load feature engineering for player fatigue."""

from __future__ import annotations

import pandas as pd

from fatigue_engine.config import PLAYER_COL


def build_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling load features using only matches before the current row."""

    out = df.copy()
    grouped = out.groupby(PLAYER_COL, group_keys=False)

    out["Minutes_Rolling_5"] = grouped["Minutes_Played"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    out["Minutes_Rolling_10"] = grouped["Minutes_Played"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )
    out["Distance_Rolling_5"] = grouped["distanceRun"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    out["Work_Rate_Rolling_5"] = grouped["kpi_work_rate"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["Full_Matches_Rolling_5"] = grouped["Minutes_Played"].transform(
        lambda s: (s.shift(1) >= 85).rolling(5, min_periods=1).sum()
    )
    out["Starts_Rolling_5"] = grouped["Minutes_Played"].transform(
        lambda s: (s.shift(1) >= 60).rolling(5, min_periods=1).sum()
    )
    return out
