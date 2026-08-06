"""
Model training for the resale-price prediction task.
Multi-model API (this is the main addition):
- build_baseline_pipeline(): Linear Regression, no tuning — the
    reference point every candidate model must beat to justify its
    added complexity.
- Three tunable candidates: ElasticNet, RandomForestRegressor, and
    XGBRegressor, each tuned via Optuna (TPE sampler) with k-fold
    cross-validation on the training set only.
- train_all_models(): trains the baseline + all three tuned
    candidates, returns everything needed to compare and pick a winner.
- select_best_model(): picks the best candidate by test-set RMSE.
- compute_shap_importance(): SHAP-based feature importance for any
    fitted pipeline, using TreeExplainer for tree models and LinearExplainer for linear models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import (
    MODEL_ARTIFACT_PATH,
    MODEL_NUMERIC_FEATURES,
    MODEL_BINARY_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    TRANSFORMED_TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    CV_SCORING,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_N_TRIALS = 30

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
def _sanitize_extension_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable/extension dtypes (Int64, Float64, boolean,
    string, category with pd.NA) to plain numpy-backed dtypes with np.nan
    for missing values.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix that may contain pandas extension-dtype columns.

    Returns
    -------
    pd.DataFrame
        Copy of X with extension dtypes converted to numpy dtypes.
    """
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_extension_array_dtype(X[col]):
            if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
                # Int64/Float64/boolean -> float64 (bools become 0.0/1.0/NaN,
                # which the binary SimpleImputer handles fine)
                X[col] = X[col].astype("float64")
            else:
                # string/category with pd.NA -> plain object with np.nan
                X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test X/y using MODEL_FEATURES.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, default=42
        The random state for reproducibility.

    Returns
    -------
    tuple
        A tuple of (X_train, X_test, y_train, y_test).  
    """
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing engineered/raw features: {missing}")

    X = _sanitize_extension_dtypes(df[MODEL_FEATURES])
    y = df[TRANSFORMED_TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used to preprocess features before modeling.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    ColumnTransformer 
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, MODEL_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, MODEL_CATEGORICAL_FEATURES),
            ("bin", binary_pipeline, MODEL_BINARY_FEATURES),
        ]
    )
    return preprocessor


def _build_pipeline_for(regressor) -> Pipeline:
    """Build a full modeling pipeline (preprocessing + regressor) for a given regressor.
    
    Parameters
    ----------
    regressor : estimator
        The regressor to use in the pipeline (e.g., RandomForestRegressor, ElasticNet, XGBRegressor).
    
    Returns
    -------
    Pipeline
        A scikit-learn Pipeline that includes preprocessing and the specified regressor.
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()), 
        ("regressor", regressor)
    ])


# ---------------------------------------------------------------------------
# Baseline: Linear Regression (no tuning)
# ---------------------------------------------------------------------------
def build_baseline_pipeline() -> Pipeline:
    """Linear Regression baseline. No hyperparameters to tune.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    Pipeline
        A scikit-learn Pipeline that includes preprocessing and a LinearRegression estimator.
    """
    return _build_pipeline_for(LinearRegression())


def train_baseline(
    df: pd.DataFrame, 
    test_size: float = TEST_SIZE, 
    random_state: int = RANDOM_STATE
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Train the Linear Regression baseline. 
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing features and target.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, default=42
        The random state for reproducibility. 
    
    Returns
    ------- 
    (pipeline, X_test, y_test)
    """
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    pipeline = build_baseline_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("Trained Linear Regression baseline on %s rows", len(X_train))
    return pipeline, X_test, y_test


# ---------------------------------------------------------------------------
# Tunable candidates + Optuna search spaces
# ---------------------------------------------------------------------------
def _elasticnet_search_space(trial: optuna.Trial) -> dict:
    """Define the hyperparameter search space for ElasticNet.
    
    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object used to suggest hyperparameters.
    
    Returns
    -------
    dict
        A dictionary containing the hyperparameters for ElasticNet.
    """
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _random_forest_search_space(trial: optuna.Trial) -> dict:
    """Define the hyperparameter search space for RandomForestRegressor.
    
    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object used to suggest hyperparameters.
    
    Returns
    -------
    dict
        A dictionary containing the hyperparameters for RandomForestRegressor.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _xgboost_search_space(trial: optuna.Trial) -> dict:
    """Define the hyperparameter search space for XGBRegressor.
    
    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object used to suggest hyperparameters.
    
    Returns
    -------
    dict
        A dictionary containing the hyperparameters for XGBRegressor.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

# Dataclass to hold model specifications for the registry
@dataclass
class ModelSpec:
    name: str                                           #* Name of the model (e.g., "elasticnet", "random_forest", "xgboost")
    estimator_cls: type                                 #* The class of the estimator (e.g., ElasticNet, RandomForestRegressor, XGBRegressor)
    search_space_fn: Callable[[optuna.Trial], dict]     #* Function that defines the hyperparameter search space for Optuna
    fixed_params: dict = field(default_factory=dict)    #* Fixed hyperparameters that are not tuned by Optuna (e.g., random_state, n_jobs)


# Registry of the three Optuna-tuned candidates. 
# Adding a new model to the comparison is just a new entry here
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "elasticnet": ModelSpec(
        "elasticnet",
        ElasticNet,
        _elasticnet_search_space,
        {"random_state": RANDOM_STATE, "max_iter": 10000},
    ),
    "random_forest": ModelSpec(
        "random_forest",
        RandomForestRegressor,
        _random_forest_search_space,
        {"random_state": RANDOM_STATE, "n_jobs": -1},
    ),
    "xgboost": ModelSpec(
        "xgboost",
        XGBRegressor,
        _xgboost_search_space,
        {"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0},
    ),
}


# ---------------------------------------------------------------------------
# Optuna-driven cross-validated search
# ---------------------------------------------------------------------------
def _cv_rmse(
    pipeline: Pipeline, 
    X: pd.DataFrame, y: pd.Series, 
    cv_folds: int, 
    random_state: int
) -> float:
    """Compute the mean CV RMSE for a given pipeline and dataset.
    
    Parameters
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline that includes preprocessing and a regressor.
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target vector.
    cv_folds : int
        The number of folds for k-fold cross-validation.
    random_state : int
        The random state for reproducibility.
        
    Returns
    -------
    The mean RMSE across the cross-validation folds.
    """        
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        pipeline, X, y, cv=kf, scoring=CV_SCORING, n_jobs=1
    )
    return float(-scores.mean()) 


def _make_objective(
    spec: ModelSpec, X: pd.DataFrame, y: pd.Series, cv_folds: int, random_state: int
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for a given model spec and dataset.
    
    Parameters
    ----------
    spec : ModelSpec
        The model specification containing the estimator class, search space function, and fixed parameters.
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target vector.
    cv_folds : int
        The number of folds for k-fold cross-validation.
    random_state : int
        The random state for reproducibility.
        
    Returns
    -------
    A callable that takes an Optuna trial and returns the mean CV RMSE for the model with the suggested hyperparameters.
    """
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna hyperparameter optimization."""
        params = spec.search_space_fn(trial)
        estimator = spec.estimator_cls(**{**spec.fixed_params, **params})
        pipeline = _build_pipeline_for(estimator)
        return _cv_rmse(pipeline, X, y, cv_folds, random_state)

    return objective


def optuna_search(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> optuna.Study:
    """Run an Optuna TPE hyperparameter search with k-fold CV for one
    candidate model from MODEL_REGISTRY. 
    
    Parameters
    ----------
    model_name : str
        The name of the model to tune (must be a key in MODEL_REGISTRY).
    X_train : pd.DataFrame
        The training feature matrix.
    y_train : pd.Series
        The training target vector.
    n_trials : int, default=30
        The number of Optuna trials to run.
    cv_folds : int, default=5
        The number of folds for k-fold cross-validation.
    random_state : int, default=42
        The random state for reproducibility. 
    
    Returns
    -------        
    The Optuna study object containing the results of the hyperparameter search.
    """
    # Validate model_name and retrieve the corresponding ModelSpec
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    
    # Create an Optuna study with TPE sampler and a fixed random state for reproducibility
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=random_state))
    # Define the objective function for the study using the model spec and training data
    objective = _make_objective(spec, X_train, y_train, cv_folds, random_state)
    # Run the optimization for the specified number of trials, suppressing the progress bar for cleaner logs
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "[%s] best CV RMSE=%.2f, params=%s", model_name, study.best_value, study.best_params
    )
    return study


def fit_best_pipeline(
    model_name: str, study: optuna.Study, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Refit a fresh pipeline on the full training set using the best
    hyperparameters found by an Optuna study.
    
    Parameters
    ----------
    model_name : str
        The name of the model to fit (must be a key in MODEL_REGISTRY).
    study : optuna.Study
        The Optuna study object containing the best hyperparameters.
    X_train : pd.DataFrame
        The training feature matrix.
    y_train : pd.Series
        The training target vector. 
        
    Returns
    -------
    Pipeline
        A scikit-learn Pipeline fitted on the full training set with the best hyperparameters.
    """
    # Validate model_name and retrieve the corresponding ModelSpec
    if model_name not in MODEL_REGISTRY: 
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name] 
    # Create a new estimator instance with the best hyperparameters
    estimator = spec.estimator_cls(**{**spec.fixed_params, **study.best_params})
    # Build a new pipeline with the best estimator and refit it on the full training data 
    pipeline = _build_pipeline_for(estimator)
    pipeline.fit(X_train, y_train)
    return pipeline

# ---------------------------------------------------------------------------
# Orchestration: baseline + all tuned candidates, then pick a winner
# ---------------------------------------------------------------------------
@dataclass
class TrainingResult:
    name: str                   # Name of the model (e.g., "linear_regression", "elasticnet", "random_forest", "xgboost")
    pipeline: Pipeline          # The fitted scikit-learn Pipeline for this model
    cv_rmse: Optional[float]    # The mean CV RMSE from the Optuna search (None for the untuned baseline)
    best_params: Optional[dict] # Best hyperparameters found by Optuna (None for the untuned baseline)


def train_all_models(
    df: pd.DataFrame,
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = CV_FOLDS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    models: Optional[list[str]] = None,
) -> tuple[dict[str, TrainingResult], pd.DataFrame, pd.Series]:
    """Train the Linear Regression baseline plus Optuna-tuned ElasticNet,
    RandomForest, and XGBoost candidates on a single shared train/test
    split, so every model is compared on identical data.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing features and target.
    n_trials : int, default=30
        The number of Optuna trials to run for each candidate model.
    cv_folds : int, default=5
        The number of folds for k-fold cross-validation during Optuna search.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, default=42
        The random state for reproducibility.
    models : list of str, optional
        An optional subset of MODEL_REGISTRY keys to tune (defaults to
        all three). Useful for quick tests / iteration.

    Returns
    -------
    (results, X_test, y_test) where results is a dict of
    {model_name: TrainingResult}. Keyed "linear_regression" for the
    baseline and by MODEL_REGISTRY key for each tuned candidate.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

    # to hold the training results for each model
    results: dict[str, TrainingResult] = {}
    
    # Train the untuned Linear Regression baseline and store its results
    baseline_pipeline = build_baseline_pipeline()
    baseline_pipeline.fit(X_train, y_train)
    results["linear_regression"] = TrainingResult(
        name="linear_regression", pipeline=baseline_pipeline, cv_rmse=None, best_params=None
    )
    
    # Train each candidate model in the registry using Optuna for hyperparameter tuning
    for model_name in models or list(MODEL_REGISTRY):
        # Run Optuna search to find the best hyperparameters for the current model
        study = optuna_search(
            model_name, X_train, y_train, n_trials=n_trials, cv_folds=cv_folds,
            random_state=random_state,
        )
        # Fit a fresh pipeline on the full training set using the best hyperparameters found by Optuna
        pipeline = fit_best_pipeline(model_name, study, X_train, y_train)
        results[model_name] = TrainingResult(
            name=model_name, pipeline=pipeline, cv_rmse=study.best_value,
            best_params=study.best_params,
        )

    return results, X_test, y_test


def select_best_model(
    results: dict[str, TrainingResult], 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> tuple[str, float]:
    """Pick the best model by held-out test-set RMSE (not CV RMSE). This is the final evaluation metric for model selection.
    
    Parameters
    ----------
    results : dict[str, TrainingResult]
        A dictionary of model names to their corresponding TrainingResult objects.
    X_test : pd.DataFrame
        The test feature matrix.
    y_test : pd.Series
        The test target vector. 
    
    Returns
    -------
    The name of the best model and its corresponding test-set RMSE.
    """
    best_name, best_rmse = None, float("inf")
    for name, result in results.items():
        # Evaluate the model on the held-out test set and compute RMSE
        preds = result.pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        if rmse < best_rmse:
            best_name, best_rmse = name, rmse
    return best_name, best_rmse


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(
    pipeline: Pipeline, 
    path: str | Path = MODEL_ARTIFACT_PATH
) -> Path:
    """Save a fitted pipeline to disk using joblib.
    
    Parameters
    ----------
    pipeline : Pipeline
        The fitted scikit-learn Pipeline to save.
    path : str or Path, default=MODEL_ARTIFACT_PATH
        The file path where the model artifact will be saved.
    
    Returns
    -------
    Path
        The path where the model artifact was saved.
    """
    # Ensure the parent directory exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save the pipeline to disk using joblib
    joblib.dump(pipeline, path)
    logger.info("Saved model artifact to %s", path)
    return path


def load_model(path: str | Path = MODEL_ARTIFACT_PATH) -> Pipeline:
    """Load a fitted pipeline from disk using joblib.

    Parameters
    ----------
    path : str or Path, default=MODEL_ARTIFACT_PATH
        The file path from which to load the model artifact.

    Returns
    -------
    Pipeline
        The loaded scikit-learn Pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model artifact found at {path}. Train a model first.")
    return joblib.load(path)


