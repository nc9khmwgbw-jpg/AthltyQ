"""Dashboard-oriented exports for Fatigue Engine outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    """Export the full Fatigue V3 dataset for downstream dashboard work."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
