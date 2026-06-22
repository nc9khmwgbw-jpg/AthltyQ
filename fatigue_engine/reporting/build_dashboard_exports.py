"""Build dashboard-ready exports from Dataset_Fatigue_V3."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "artifacts" / "datasets" / "dataset_fatigue_v3.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "dashboard"
MIN_VALID_SCORES_FOR_RELIABLE_SUMMARY = 5

PLAYER_COL = "Nom"
DATE_COL = "Match_Date"
SCORE_COL = "AthltyQ_Fatigue_Index"
MEDICAL_COL = "Medical_Fragility_Score"
LEVEL_COL = "Fatigue_Level"

EXPORT_COLUMNS = [
    "Nom",
    "Player_Name",
    "Team",
    "League",
    "Position",
    "Age",
    "Match_Date",
    "Next_Match_Date",
    "Fatigue_Level",
    "Primary_Fatigue_Driver",
    "AthltyQ_Fatigue_Index",
    "Charge_Score",
    "Recovery_Score",
    "Intensity_Score",
    "Medical_Fragility_Score",
    "Medical_Data_Quality_Flag",
]

logger = logging.getLogger(__name__)


def _round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric columns for compact dashboard files."""

    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(3)
    return out


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Fatigue V3 dataset not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if "Next_Match_Date" in df.columns:
        df["Next_Match_Date"] = pd.to_datetime(df["Next_Match_Date"], errors="coerce")

    for column in [
        SCORE_COL,
        MEDICAL_COL,
        "Charge_Score",
        "Recovery_Score",
        "Intensity_Score",
        "Age",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values([PLAYER_COL, DATE_COL]).reset_index(drop=True)


def build_latest_players(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest dashboard-usable observation for each player.

    The raw last match for a player often has no Next_Match_Date, which makes
    Recovery_Score and the final index unavailable. Dashboard monitoring needs
    the latest row where the fatigue index is actually computed.
    """

    latest = df.dropna(subset=[PLAYER_COL, DATE_COL, SCORE_COL]).sort_values([PLAYER_COL, DATE_COL])
    latest = latest.groupby(PLAYER_COL, as_index=False).tail(1)
    return latest.sort_values(SCORE_COL, ascending=False, na_position="last").reset_index(drop=True)


def _alert_export(latest_players: pd.DataFrame, level: str) -> pd.DataFrame:
    alert = latest_players[latest_players[LEVEL_COL].eq(level)].copy()
    return alert.sort_values(SCORE_COL, ascending=False, na_position="last")


def _summary_by_group(latest_players: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = (
        latest_players.groupby(group_col, dropna=False)
        .agg(
            players_count=(PLAYER_COL, "nunique"),
            valid_scores_count=(SCORE_COL, "count"),
            avg_fatigue_score=(SCORE_COL, "mean"),
            median_fatigue_score=(SCORE_COL, "median"),
            max_fatigue_score=(SCORE_COL, "max"),
            red_alerts=(LEVEL_COL, lambda s: int(s.eq("red").sum())),
            orange_alerts=(LEVEL_COL, lambda s: int(s.eq("orange").sum())),
            yellow_alerts=(LEVEL_COL, lambda s: int(s.eq("yellow").sum())),
            alert_players_count=(LEVEL_COL, lambda s: int(s.isin(["red", "orange", "yellow"]).sum())),
            alert_rate=(LEVEL_COL, lambda s: float(s.isin(["red", "orange", "yellow"]).mean())),
        )
        .reset_index()
    )
    summary["alert_rate"] = (summary["alert_rate"] * 100).round(2)
    summary["reliable_summary"] = summary["valid_scores_count"] >= MIN_VALID_SCORES_FOR_RELIABLE_SUMMARY
    return summary.sort_values(["avg_fatigue_score", "players_count"], ascending=[False, False])


def _top_players(latest_players: pd.DataFrame, score_col: str, n: int = 20) -> pd.DataFrame:
    return latest_players.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(n)


def _dashboard_metrics(latest_players: pd.DataFrame, source_rows: int) -> dict[str, Any]:
    levels = latest_players[LEVEL_COL].value_counts(dropna=False).to_dict()
    teams_summary = _summary_by_group(latest_players, "Team")
    leagues_summary = _summary_by_group(latest_players, "League")
    return {
        "source_rows": int(source_rows),
        "total_players": int(latest_players[PLAYER_COL].nunique()),
        "teams": int(latest_players["Team"].nunique()) if "Team" in latest_players.columns else 0,
        "leagues": int(latest_players["League"].nunique()) if "League" in latest_players.columns else 0,
        "reliable_teams_count": int(teams_summary["reliable_summary"].sum()),
        "unreliable_teams_count": int((~teams_summary["reliable_summary"]).sum()),
        "reliable_leagues_count": int(leagues_summary["reliable_summary"].sum()),
        "unreliable_leagues_count": int((~leagues_summary["reliable_summary"]).sum()),
        "alerts": {
            "red": int(levels.get("red", 0)),
            "orange": int(levels.get("orange", 0)),
            "yellow": int(levels.get("yellow", 0)),
            "normal": int(levels.get("normal", 0)),
        },
        "global_average_fatigue_score": round(float(latest_players[SCORE_COL].mean()), 3),
        "missing_fatigue_score_players": int(latest_players[SCORE_COL].isna().sum()),
        "latest_selection_rule": "latest row per player with non-null AthltyQ_Fatigue_Index",
    }


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> Path:
    out = df.copy()
    if columns is not None:
        out = out[_available_columns(out, columns)]
    for column in [DATE_COL, "Next_Match_Date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    out = _round_numeric(out)
    out.to_csv(path, index=False)
    return path


def build_dashboard_exports(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Create all dashboard CSV and JSON exports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_dataset(input_path)
    latest = build_latest_players(df)
    teams_summary = _summary_by_group(latest, "Team")
    leagues_summary = _summary_by_group(latest, "League")
    top_teams_risk = teams_summary[teams_summary["reliable_summary"]].sort_values(
        ["alert_rate", "avg_fatigue_score", "players_count"],
        ascending=[False, False, False],
    )

    paths = {
        "latest_players": _write_csv(latest, output_dir / "latest_players.csv", EXPORT_COLUMNS),
        "alerts_red": _write_csv(_alert_export(latest, "red"), output_dir / "alerts_red.csv", EXPORT_COLUMNS),
        "alerts_orange": _write_csv(
            _alert_export(latest, "orange"), output_dir / "alerts_orange.csv", EXPORT_COLUMNS
        ),
        "alerts_yellow": _write_csv(
            _alert_export(latest, "yellow"), output_dir / "alerts_yellow.csv", EXPORT_COLUMNS
        ),
        "teams_summary": _write_csv(teams_summary, output_dir / "teams_summary.csv"),
        "leagues_summary": _write_csv(leagues_summary, output_dir / "leagues_summary.csv"),
        "top_teams_risk": _write_csv(top_teams_risk, output_dir / "top_teams_risk.csv"),
        "top20_fatigue": _write_csv(
            _top_players(latest, SCORE_COL, 20), output_dir / "top20_fatigue.csv", EXPORT_COLUMNS
        ),
        "top20_medical_fragility": _write_csv(
            _top_players(latest, MEDICAL_COL, 20),
            output_dir / "top20_medical_fragility.csv",
            EXPORT_COLUMNS,
        ),
    }

    metrics_path = output_dir / "dashboard_metrics.json"
    metrics_path.write_text(
        json.dumps(_dashboard_metrics(latest, len(df)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["dashboard_metrics"] = metrics_path

    logger.info("Dashboard exports written to %s", output_dir)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    paths = build_dashboard_exports()
    for name, path in paths.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
