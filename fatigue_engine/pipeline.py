"""End-to-end AthltyQ V3 Fatigue Engine pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging

import pandas as pd

from fatigue_engine.config import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH
from fatigue_engine.features import build_fatigue_features
from fatigue_engine.normalization import normalize_fatigue_features
from fatigue_engine.reporting import export_dataset
from fatigue_engine.scoring import add_fatigue_index
from fatigue_engine.validation.leakage_validator import (
    drop_leakage_columns,
    validate_no_leakage_columns,
)
from fatigue_engine.validation.quality_checks import build_quality_summary, validate_score_bounds
from fatigue_engine.validation.schema_validator import validate_input_schema
from fatigue_engine.validation.temporal_validator import (
    validate_rolling_features_have_history,
    validate_temporal_order,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the Fatigue Engine and export Dataset_Fatigue_V3."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger.info("Loading fatigue input dataset: %s", input_path)
    raw = pd.read_csv(input_path)
    validate_input_schema(raw)

    logger.info("Building fatigue features")
    features = build_fatigue_features(raw)
    validate_temporal_order(features)
    validate_rolling_features_have_history(features)

    logger.info("Normalizing fatigue features")
    normalized = normalize_fatigue_features(features)

    logger.info("Calculating fatigue scores")
    scored = add_fatigue_index(normalized)
    validate_score_bounds(scored)

    logger.info("Dropping leakage columns before export")
    final = drop_leakage_columns(scored)
    validate_no_leakage_columns(final)

    summary = build_quality_summary(final)
    logger.info("Fatigue dataset quality summary: %s", summary)

    logger.info("Exporting Dataset_Fatigue_V3: %s", output_path)
    export_dataset(final, output_path)
    return final


if __name__ == "__main__":
    run_pipeline()
