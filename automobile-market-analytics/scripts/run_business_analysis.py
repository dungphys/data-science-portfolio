"""
Entry point: run all business-question analyses and write a Markdown
report to reports/business_report.md.

Requires a trained model artifact (run scripts/run_training.py first)
so that Q1 (top price drivers) can use real feature importances.

"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from src.config import BUSINESS_REPORT_PATH, MODEL_ARTIFACT_PATH, RAW_DATA_PATH
from src.business.business_questions import run_all_business_questions
from src.data.loader import load_and_clean
from src.features.build_features import build_features
from src.models.evaluate import get_feature_importance
from src.models.train import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

QUESTION_TEXT = {
    "q1_top_price_drivers": "Q1. Which vehicle attributes most influence resale price?",
    "q2_accident_price_penalty": "Q2. How much does a prior accident reduce resale value?",
    "q3_depreciation_by_age": "Q3. What does the depreciation curve look like as vehicles age?",
    "q4_price_by_fuel_and_body": "Q4. Which fuel type / body type combinations command the highest resale value?",
    "q5_service_history_value": "Q5. Does a fuller service history translate into measurable resale value?",
    "q6_regional_price_variation": "Q6. Which states show the highest average resale prices?",
    "q7_mileage_efficiency_relationship": "Q7. Does annualized mileage correlate with lower resale price?",
}


def build_markdown_report(results: dict) -> str:
    lines = [
        "# Business Analysis Report — Automotive Resale Market",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "This report answers the business questions defined in the project ",
        "README using the cleaned dataset and the trained resale-price model. ",
        "See README.md for full methodology notes.",
        "",
    ]
    for key, question in QUESTION_TEXT.items():
        lines.append(f"## {question}")
        lines.append("")
        result = results[key]
        lines.append("```")
        lines.append(result.to_string())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main(data_path: str = str(RAW_DATA_PATH)) -> None:
    logger.info("Loading and cleaning data from %s", data_path)
    df = load_and_clean(data_path)
    df = build_features(df)

    logger.info("Loading trained model from %s", MODEL_ARTIFACT_PATH)
    pipeline = load_model(MODEL_ARTIFACT_PATH)
    importance = get_feature_importance(pipeline)

    logger.info("Running business question analyses")
    results = run_all_business_questions(df, importance)

    report_md = build_markdown_report(results)
    BUSINESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUSINESS_REPORT_PATH, "w") as f:
        f.write(report_md)
    logger.info("Business report written to %s", BUSINESS_REPORT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run business question analysis.")
    parser.add_argument("--data-path", default=str(RAW_DATA_PATH))
    args = parser.parse_args()
    main(args.data_path)