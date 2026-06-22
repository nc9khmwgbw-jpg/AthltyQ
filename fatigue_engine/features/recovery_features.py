"""Recovery and calendar feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fatigue_engine.config import DATE_COL, PLAYER_COL


def _count_matches_last_days(player_df: pd.DataFrame, days: int) -> pd.Series:
    dates = player_df[DATE_COL].to_numpy(dtype="datetime64[ns]")
    window = np.timedelta64(days, "D")
    counts = [int(((dates < date) & (dates >= date - window)).sum()) for date in dates]
    return pd.Series(counts, index=player_df.index)


def build_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest and calendar-density features."""

    out = df.copy()
    grouped = out.groupby(PLAYER_COL, group_keys=False)

    out["Prev_Match_Date"] = grouped[DATE_COL].shift(1)
    out["Next_Match_Date"] = grouped[DATE_COL].shift(-1)
    out["Days_Rest_Before_T"] = (out[DATE_COL] - out["Prev_Match_Date"]).dt.days
    out["Days_Rest_Before_T1"] = (out["Next_Match_Date"] - out[DATE_COL]).dt.days

    for days in (7, 14, 30):
        out[f"Matches_Last_{days}D"] = grouped.apply(
            lambda player_df, d=days: _count_matches_last_days(player_df, d)
        ).reset_index(level=0, drop=True)

    out["Short_Rest_Flag"] = (out["Days_Rest_Before_T1"] <= 3).astype(float) * 100.0
    out["Congested_Period_Flag"] = (out["Matches_Last_14D"] >= 4).astype(float) * 100.0
    return out
