"""
Central configuration: paths, column schema, and constants used across
the pipeline (EDA, feature engineering, modeling, business analysis).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATAFILE_NAME = "automobile_dataset.csv"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / DATAFILE_NAME

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_ARTIFACT_PATH = MODELS_DIR / "price_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
BUSINESS_REPORT_PATH = RESULTS_DIR / "business_report.md"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
UNTRANSFORMED_TARGET = "Selling_Price"
TRANSFORMED_TARGET = "Selling_Price_Log" # used for modeling after log transformation / created in feature engineering

NUMERIC_FEATURES = [
    "Year",
    "Engine_Size",
    "Mileage",
    "Horsepower",
    "Torque",
    "Owners",
    "Fuel_Efficiency",
]

CATEGORICAL_FEATURES = [
    "Make",
    "Model",
    "Fuel_Type",
    "Transmission",
    "Service_History",
    "Color",
    "Body_Type",
    "Drivetrain",
    "Location",
]

BINARY_FEATURES = ["Accident_History"]

ALL_RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Engineered features added by src/features/build_features.py
ENGINEERED_FEATURES = ["Vehicle_Age", "Mileage_Per_Year", "Power_Per_Liter"]

# Features to remove from the dataset before modeling (e.g. high-cardinality categorical features, collinearity, etc.)
FEATURES_REMOVAL = [
    "Model", # high-cardinality categorical feature removed to avoid excessive dimensionality
    "Year",  # collinear with Vehicle_Age (Year is used to compute Vehicle_Age, which is a more useful feature)
    "Location", # not so correlated to the price
]

# Features entering the modeling (raw + engineered features, excluding removed features)
MODEL_NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES + ENGINEERED_FEATURES if f not in FEATURES_REMOVAL]
MODEL_CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f not in FEATURES_REMOVAL]
MODEL_BINARY_FEATURES = [f for f in BINARY_FEATURES if f not in FEATURES_REMOVAL]
MODEL_FEATURES = [f for f in ALL_RAW_FEATURES + ENGINEERED_FEATURES if f not in FEATURES_REMOVAL]

# parameters for train/test split and cross-validation
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_SCORING = "neg_root_mean_squared_error"  # RMSE is the primary metric for model evaluation
# ---------------------------------------------------------------------------
# evaluation metrics for regression models (used in cross-validation and final evaluation)
OTHER_CV_SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}

# Current year used to compute vehicle age. Kept as a constant (rather than
# datetime.now()) so feature engineering is deterministic and testable.
# for dynamic current year: CURRENT_YEAR = datetime.datetime.now().year (import datetime)
CURRENT_YEAR = 2026

# just a test run to check if config.py is loaded correctly
if __name__ == "__main__":
    print(f"Configuration loaded. Root directory: {ROOT_DIR}")
    print(ALL_RAW_FEATURES)