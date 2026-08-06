"""
Entry point: run exploratory data analysis end-to-end.
"""
from __future__ import annotations

import argparse
import logging

from src.config import FIGURES_DIR, RAW_DATA_PATH, RESULTS_DIR
from src.data.loader import load_and_clean
from src.eda.eda import (
    correlation_with_target,
    missing_value_summary,
    numeric_summary,
    price_by_category,
    outlier_bounds_iqr,
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_scatter_relationship,
    plot_boxplot_by_category,
    plot_mid_target_vs_category,
    generate_all_plots
)
from src.features.build_features import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(data_path: str = str(RAW_DATA_PATH)) -> None:
    """Run EDA on the automotive dataset 
    and save results to results/eda_summary.txt and figures/.
    
    Parameters
    ----------
    data_path : str
        Path to the raw automotive dataset CSV file.    
    
    """
    # Load and clean the data, then build features
    logger.info("Loading and cleaning data from %s", data_path)
    df = load_and_clean(data_path)
    df = build_features(df)

    # Basic EDA: missing values, numeric summary, correlation with target,
    # median/mean/count by typical categories
    logger.info("=== Missing Value Summary ===\n%s", missing_value_summary(df))
    logger.info("=== Numeric Summary ===\n%s", numeric_summary(df))
    logger.info("=== Correlation with Selling_Price ===\n%s", correlation_with_target(df))
    logger.info("=== Median/Mean/Count by Typical Categories ===\n")
    cat_stats = ["Make", "Body_Type", 
                "Fuel_Type", "Transmission", 
                "Owners", "Accident_History"]
    for cat in cat_stats:
        logger.info("=== Price Stats by %s ===\n%s", cat, price_by_category(df, cat))
    
    # generate all plots
    paths = generate_all_plots(df, FIGURES_DIR)
    logger.info("EDA complete. Figures saved:\n%s", "\n".join(str(p) for p in paths))

    summary_path = RESULTS_DIR / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write("MISSING VALUES\n")
        f.write(missing_value_summary(df).to_string())
        f.write("\n\nNUMERIC SUMMARY\n")
        f.write(numeric_summary(df).to_string())
        f.write("\n\nCORRELATION WITH SELLING_PRICE\n")
        f.write(correlation_with_target(df).to_string())
        for cat in cat_stats:
            f.write(f"\n\nMEDIAN PRICE BY {cat}\n")
            f.write(price_by_category(df, cat).to_string())
    logger.info("Text summary written to %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on the automotive dataset.")
    parser.add_argument("--data-path", default=str(RAW_DATA_PATH))
    args = parser.parse_args()
    main(args.data_path)