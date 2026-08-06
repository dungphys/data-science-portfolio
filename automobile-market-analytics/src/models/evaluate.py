"""
Evaluation utilities: regression metrics and feature importance
extraction from a fitted pipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path


import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from xgboost import XGBRegressor

import matplotlib.pyplot as plt 

from src.config import METRICS_PATH, RANDOM_STATE

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: pd.Series, 
    y_pred: np.ndarray
) -> dict:
    """Compute standard regression metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    
    Returns
    -------
    dict
        Dictionary containing RMSE, MAE, R2, and MAPE metrics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {
        "rmse": rmse, 
        "mae": mae, 
        "r2": r2, 
        "mape": mape
    }


def evaluate_pipeline(
    pipeline: Pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> dict:
    """Evaluate a fitted pipeline on test data and compute regression metrics.
    
    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline that includes preprocessing and a regressor.
    X_test : pd.DataFrame
        The feature matrix for the test set.
    y_test : pd.Series
        The true target values for the test set.
        
    Returns
    -------
    dict
        Dictionary containing RMSE, MAE, R2, and MAPE metrics.
    """
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract post-preprocessing feature names (numeric + one-hot expanded).
    
    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline that includes preprocessing and a regressor.
    
    Returns
    -------
    list[str]
        List of feature names after preprocessing (including one-hot encoded categorical features).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def get_feature_importance(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Return a DataFrame of feature importance sorted descending.
    
    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline that includes preprocessing and a regressor.
    top_n : int, default=20
        The number of top features to return.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names and their importance.
    """
    feature_names = get_feature_names(pipeline)
    regressor = pipeline.named_steps["regressor"]
    importances = regressor.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)

# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------
def _get_shap_explainer(regressor, background_data: np.ndarray) -> shap.Explainer:
    """Pick an appropriate SHAP explainer for the regressor's family.

    TreeExplainer is exact and fast for tree ensembles (RandomForest,
    XGBoost). LinearExplainer is exact and fast for linear models
    (LinearRegression, ElasticNet). Anything else falls back to the
    generic (slower, sampling-based) shap.Explainer.
    
    Parameters
    ----------
    regressor : estimator
        The fitted regressor for which to compute SHAP values.
    background_data : np.ndarray
        A representative sample of the training data (post-preprocessing) to use as background for SHAP value computation. 
        
    Returns
    -------
    shap.Explainer
        An appropriate SHAP explainer instance for the given regressor.
    """
    if isinstance(regressor, (RandomForestRegressor, XGBRegressor)):
        return shap.TreeExplainer(regressor)
    if isinstance(regressor, (LinearRegression, ElasticNet)):
        return shap.LinearExplainer(regressor, background_data)
    return shap.Explainer(regressor, background_data)


def compute_shap_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    sample_size: int = 500,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute mean |SHAP value| feature importance for a fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline that includes preprocessing and a regressor.
    X : pd.DataFrame
        The feature matrix (preprocessed) for which to compute SHAP values.
    sample_size : int, default=200
        The number of rows to subsample from X for SHAP value computation   
    random_state : int, default=42
        The random state for reproducibility when subsampling rows.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame containing features and their corresponding mean absolute SHAP values, sorted in descending order
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    # Subsample the data if it exceeds the specified sample size for faster SHAP computation
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=random_state)

    # Transform the features using the preprocessor to get the post-preprocessing feature matrix
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = list(preprocessor.get_feature_names_out())
    
    # Keep a DataFrame version around for the beeswarm plot's color axis
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    # Compute SHAP values using the appropriate explainer based on the regressor type
    explainer = _get_shap_explainer(regressor, X_transformed)
    if hasattr(explainer, "shap_values"):
        shap_values = explainer.shap_values(X_transformed)
    else:
        shap_values = explainer(X_transformed).values
        
    # Compute mean absolute SHAP values for each feature and return a sorted DataFrame
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names, 
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap
    })
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance, shap_values, X_transformed_df


def save_metrics(metrics: dict, path: str | Path = METRICS_PATH) -> Path:
    """Save evaluation metrics to a JSON file.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics.
    path : str | Path, default=METRICS_PATH
        Path to save the metrics JSON file. 
    
    Returns
    -------
    Path
        Path to the saved metrics JSON file.
    """
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", path)
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_shap_importance(
    importance: pd.DataFrame,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 20,
    path: str | Path = "results/figures/shap_importance.png",
    model_name: str | None = None,
) -> Path:
    df = importance.head(top_n).iloc[::-1]
    top_features = df["feature"].tolist()

    fig, ax = plt.subplots(1, 2, figsize=(16, max(4, 0.35 * len(df))))
    ax[0].barh(df["feature"], df["mean_abs_shap"], color="#eb23c0")
    ax[0].set_xlabel("Mean |SHAP value|")

    feature_idx = [X.columns.get_loc(f) for f in top_features]
    cmap = plt.get_cmap("coolwarm")

    for row, fi in enumerate(feature_idx):
        vals = shap_values[:, fi]
        fvals = X.iloc[:, fi].values

        vmin, vmax = np.percentile(fvals, [5, 95])
        vmin, vmax = (vmin, vmax) if vmax > vmin else (fvals.min(), fvals.max() + 1e-9)
        norm_fvals = np.clip((fvals - vmin) / (vmax - vmin + 1e-9), 0, 1)

        y_jitter = row + (np.random.rand(len(vals)) - 0.5) * 0.6
        ax[1].scatter(vals, y_jitter, c=norm_fvals, cmap=cmap, s=8, alpha=0.7, linewidths=0)

    ax[1].set_yticks(range(len(top_features)))
    ax[1].set_yticklabels(top_features)
    ax[1].axvline(0, color="black", linewidth=0.8)
    ax[1].set_xlabel("SHAP value (signed, per sample)")
    ax[1].set_ylim(-0.5, len(top_features) - 0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value")

    title = "SHAP Feature Importance"
    if model_name:
        title += f" ({model_name})"
    fig.suptitle(title)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved SHAP importance plot to %s", path)
    return path


def plot_model_comparison(
    results: dict[str, "TrainingResult"],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: str | Path = "results/figures/model_comparison.png",
) -> Path:
    """Bar plot comparing held-out test-set RMSE across all trained
    candidates, highlighting the best model.

    Parameters
    ----------
    results : dict[str, TrainingResult]
        Mapping of model name to TrainingResult, as returned by
        train_all_models() (each result must expose a fitted `.pipeline`).
    X_test : pd.DataFrame
        The test feature matrix.
    y_test : pd.Series
        The test target vector.
    path : str | Path, default="reports/figures/model_comparison.png"
        Where to save the PNG.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    names, rmses = [], []
    for name, result in results.items():
        preds = result.pipeline.predict(X_test)
        rmses.append(float(np.sqrt(mean_squared_error(y_test, preds))))
        names.append(name)

    order = np.argsort(rmses)
    names = [names[i] for i in order]
    rmses = [rmses[i] for i in order]
    colors = ["#55A868" if i == 0 else "#4C72B0" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, rmses, color=colors)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Model Comparison — Test RMSE (lower is better)")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.margins(y=0.15)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison plot to %s", path)
    return path