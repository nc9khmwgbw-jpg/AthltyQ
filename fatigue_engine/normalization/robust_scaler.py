"""Robust feature scaling helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def robust_score(
    df: pd.DataFrame,
    column: str,
    group_cols: Sequence[str],
    *,
    inverse: bool = False,
    min_group_size: int = 100,
) -> pd.Series:
    """Scale a numeric feature to 0-100 using robust grouped z-scores."""

    values = pd.to_numeric(df[column], errors="coerce")
    global_median = values.median()
    global_iqr = values.quantile(0.75) - values.quantile(0.25)
    if not np.isfinite(global_iqr) or global_iqr == 0:
        global_iqr = 1.0

    scaled = pd.Series(index=df.index, dtype=float)
    existing_group_cols = [col for col in group_cols if col in df.columns]
    groups = df.groupby(existing_group_cols, dropna=False) if existing_group_cols else [(None, df)]

    for _, group in groups:
        idx = group.index
        group_values = values.loc[idx].dropna()
        if len(group_values) >= min_group_size:
            median = group_values.median()
            iqr = group_values.quantile(0.75) - group_values.quantile(0.25)
        else:
            median = global_median
            iqr = global_iqr

        if not np.isfinite(iqr) or iqr == 0:
            iqr = global_iqr

        scaled.loc[idx] = 50.0 + 15.0 * ((values.loc[idx] - median) / iqr)

    scaled = scaled.clip(0.0, 100.0)
    if inverse:
        scaled = 100.0 - scaled
    return scaled
