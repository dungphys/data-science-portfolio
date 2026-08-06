"""
Entry point: train the resale-price prediction model end-to-end and
persist the model artifact + metrics.

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error

from src.config import (
    FIGURES_DIR,
    MODEL_FEATURES,
    MODEL_ARTIFACT_PATH, 
    METRICS_PATH, 
    RAW_DATA_PATH,
    CV_FOLDS
)
from src.data.loader import load_and_clean
from src.features.build_features import build_features
from src.models.evaluate import (
    compute_shap_importance, 
    evaluate_pipeline, 
    save_metrics,
    plot_model_comparison,
    plot_shap_importance
)
from src.models.train import (
    DEFAULT_N_TRIALS, 
    MODEL_REGISTRY, 
    save_model, 
    select_best_model, 
    train_all_models
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(
    data_path: str = str(RAW_DATA_PATH),
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = CV_FOLDS,
    models: list[str] | None = None,
) -> None:
    logger.info("Loading and cleaning data from %s", data_path)
    df = load_and_clean(data_path)
    logger.info("Feature Engineering")
    df = build_features(df)
    print(df[MODEL_FEATURES].dtypes)

    logger.info(
        "Training baseline + tuned candidates %s (n_trials=%s, cv_folds=%s)",
        models or list(MODEL_REGISTRY),
        n_trials,
        cv_folds,
    )
    results, X_test, y_test = train_all_models(
        df, n_trials=n_trials, cv_folds=cv_folds, models=models
    )
    
    all_rmses = {
        name: float(np.sqrt(mean_squared_error(y_test, result.pipeline.predict(X_test))))
        for name, result in results.items()
    }
    for name, rmse in sorted(all_rmses.items(), key=lambda kv: kv[1]):
        logger.info("[%s] test RMSE=%.4f", name, rmse)
    logger.info("Linear regression baseline test RMSE=%.4f", all_rmses["linear_regression"])

    best_name, best_rmse = select_best_model(results, X_test, y_test)
    logger.info("Best model: %s (test RMSE=%.4f)", best_name, best_rmse)
    best_pipeline = results[best_name].pipeline
    
    plot_model_comparison(
        results, 
        X_test, y_test, 
        path=FIGURES_DIR / "model_comparison.png"
    )

    metrics = evaluate_pipeline(best_pipeline, X_test, y_test)
    logger.info("Test set metrics for %s: %s", best_name, metrics)
    save_metrics(metrics, METRICS_PATH)
    
    importance, shap_values, X_transformed_df = compute_shap_importance(best_pipeline, X_test)
    logger.info("Top SHAP feature importances for %s:\n%s", best_name, importance.head(20).to_string())
    plot_shap_importance(importance, shap_values, X_transformed_df, path=FIGURES_DIR / "shap_importance.png", model_name=best_name)

    save_model(best_pipeline, MODEL_ARTIFACT_PATH)
    logger.info("Training complete. Best model (%s) saved to %s", best_name, MODEL_ARTIFACT_PATH)
    logger.info("Figures saved to %s", FIGURES_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and select the best resale price model.")
    parser.add_argument("--data-path", default=str(RAW_DATA_PATH))
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Optuna trials per candidate model")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS, help="K-fold CV folds used during Optuna search")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY),
        default=None,
        help="Subset of tunable candidates to train (default: all). Baseline linear regression always runs.",
    )
    args = parser.parse_args()
    main(args.data_path, args.n_trials, args.cv_folds, args.models)