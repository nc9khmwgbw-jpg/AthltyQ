"""Final AthltyQ fatigue index calculation."""

from __future__ import annotations

import pandas as pd

from fatigue_engine.config import ALERT_PERCENTILES, FATIGUE_WEIGHTS
from fatigue_engine.scoring.charge_score import add_charge_score
from fatigue_engine.scoring.intensity_score import add_intensity_score
from fatigue_engine.scoring.medical_fragility_score import add_medical_fragility_score
from fatigue_engine.scoring.recovery_score import add_recovery_score


def _driver(row: pd.Series) -> str:
    drivers = {
        "Charge": row["Charge_Score"],
        "Recovery": row["Recovery_Score"],
        "Intensity": row["Intensity_Score"],
        "Medical": row["Medical_Fragility_Score"],
    }
    return max(drivers, key=drivers.get)


def _level(score: float, yellow: float, orange: float, red: float) -> str:
    if score >= red:
        return "red"
    if score >= orange:
        return "orange"
    if score >= yellow:
        return "yellow"
    return "normal"


def add_fatigue_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add all sub-scores, final index, alert thresholds, and driver labels."""

    out = add_charge_score(df)
    out = add_recovery_score(out)
    out = add_intensity_score(out)
    out = add_medical_fragility_score(out)

    out["AthltyQ_Fatigue_Index"] = (
        FATIGUE_WEIGHTS.charge * out["Charge_Score"]
        + FATIGUE_WEIGHTS.recovery * out["Recovery_Score"]
        + FATIGUE_WEIGHTS.intensity * out["Intensity_Score"]
        + FATIGUE_WEIGHTS.medical * out["Medical_Fragility_Score"]
    ).clip(0.0, 100.0)

    yellow = out["AthltyQ_Fatigue_Index"].quantile(ALERT_PERCENTILES.yellow)
    orange = out["AthltyQ_Fatigue_Index"].quantile(ALERT_PERCENTILES.orange)
    red = out["AthltyQ_Fatigue_Index"].quantile(ALERT_PERCENTILES.red)

    out["Fatigue_Alert_Yellow_Threshold"] = yellow
    out["Fatigue_Alert_Orange_Threshold"] = orange
    out["Fatigue_Alert_Red_Threshold"] = red
    out["Fatigue_Level"] = out["AthltyQ_Fatigue_Index"].apply(
        lambda score: _level(score, yellow, orange, red)
    )
    out["Primary_Fatigue_Driver"] = out.apply(_driver, axis=1)
    return out
