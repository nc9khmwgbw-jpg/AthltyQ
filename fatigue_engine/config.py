"""Configuration for the AthltyQ V3 fatigue engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "DATA_PIPELINE" / "NETTOYAGE" / "data" / "merged_dataset_clean.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "datasets" / "dataset_fatigue_v3.csv"
DEFAULT_INJURY_HISTORY_PATH = (
    PROJECT_ROOT
    / "DATA_PIPELINE"
    / "SCRAPPING"
    / "data"
    / "raw"
    / "transfermarkt"
    / "injury_history.csv"
)
DEFAULT_POSITION_REPORT_PATH = PROJECT_ROOT / "reports" / "position_mapping_report.md"

PLAYER_COL = "Nom"
DATE_COL = "Match_Date"
GROUP_COLS = ("League", "Position")

REQUIRED_COLUMNS = {
    "Nom",
    "Match_Date",
    "League",
    "Team",
    "Age",
    "Minutes_Played",
    "distanceRun",
    "sprints",
    "kpi_work_rate",
    "kpi_explosivity",
    "Tackles",
    "Interceptions",
    "Clearances",
    "Ball_Recovery",
}

LEAKAGE_COLUMNS = {
    "Target_Injury_Next_30D",
    "Next_Injury_Date",
    "Next_Injury_Type",
    "Next_Injury_Category",
    "Next_Injury_Duration_Days",
    "Days_To_Next_Injury",
    "Medical_Risk_Score",
    "Target_Fatigue",
}

MEDICAL_COLUMNS = {
    "Injury_History_Available",
    "Injury_Count_Career_Before_T",
    "Injury_Count_12M_Before_T",
    "Days_Since_Last_Injury",
    "Total_Injury_Days_12M",
    "Muscle_Injury_Count_12M",
    "Recurring_Same_Category_12M",
    "Last_Injury_Duration_Days",
}


@dataclass(frozen=True)
class FatigueWeights:
    """Top-level AthltyQ fatigue index weights."""

    charge: float = 0.30
    recovery: float = 0.30
    intensity: float = 0.25
    medical: float = 0.15


@dataclass(frozen=True)
class AlertPercentiles:
    """Dashboard alert thresholds based on score percentiles."""

    yellow: float = 0.75
    orange: float = 0.90
    red: float = 0.95


FATIGUE_WEIGHTS = FatigueWeights()
ALERT_PERCENTILES = AlertPercentiles()

POSITIVE_NORMALIZATION_FEATURES = {
    "Minutes_Rolling_5",
    "Minutes_Rolling_10",
    "Distance_Rolling_5",
    "Full_Matches_Rolling_5",
    "Work_Rate_Rolling_5",
    "Matches_Last_14D",
    "Matches_Last_7D",
    "High_Intensity_Load_5",
    "Sprints_Per_90_Rolling_5",
    "Explosivity_Rolling_5",
    "Distance_Per_Min_Rolling_5",
    "Defensive_Actions_Rolling_5",
    "Injury_Count_12M_Before_T",
    "Total_Injury_Days_12M",
    "Muscle_Injury_Count_12M",
    "Last_Injury_Duration_Days",
}

INVERSE_NORMALIZATION_FEATURES = {
    "Days_Rest_Before_T",
    "Days_Rest_Before_T1",
    "Days_Since_Last_Injury",
}
