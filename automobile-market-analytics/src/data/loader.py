"""
Loading and cleaning the raw automobile market dataset.

To do:
- Read the CSV from disk.
- Validate the expected schema is present.
- Apply light, well-defined cleaning (dtype coercion, dedup, basic
sanity filters) without doing feature engineering.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ALL_RAW_FEATURES, CURRENT_YEAR, RAW_DATA_PATH, DATAFILE_NAME,
    NUMERIC_FEATURES, UNTRANSFORMED_TARGET, BINARY_FEATURES, CATEGORICAL_FEATURES
)

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = ALL_RAW_FEATURES + [UNTRANSFORMED_TARGET]


class SchemaError(ValueError):
    """Raised when the input dataframe does not match the expected schema."""


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file into a DataFrame.

    Parameters
    ----------
    path : str | Path
        Location of the CSV file. Defaults to data/automobile_dataset.csv.
        
    Returns
    -------
    pd.DataFrame
        The raw dataset as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place {DATAFILE_NAME} in the "
            "data/ directory before running the pipeline."
        )
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %s rows, %s columns", len(df), df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure all expected columns are present in the dataframe.

    Extra/unknown columns are tolerated (a warning is logged); missing
    required columns raise a SchemaError so failures are caught early.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to validate.
        
    Raises
    ------
    SchemaError
        If any expected columns are missing from the dataframe.
    Warning
        If any unexpected extra columns are present in the dataframe.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(f"Dataset is missing required columns: {sorted(missing)}")

    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        logger.warning("Dataset has unexpected extra columns: %s", sorted(extra))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning steps.

    - Drop exact duplicate rows.
    - Coerce numeric columns to numeric dtype (invalid parses -> NaN).
    - Drop rows with a missing/non-positive target (can't train or evaluate on those).
    - Drop rows with a negative mileage/year that indicate bad data.
    - Reset the index.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to clean.
        
    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.
    """
    df = df.copy()
    # -- Drop exact duplicates
    size_before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %s duplicate rows", size_before - len(df))
    
    # -- Coerce numeric columns to numeric dtype
    numeric_cols = NUMERIC_FEATURES + BINARY_FEATURES + [UNTRANSFORMED_TARGET]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # -- Drop rows with missing/invalid target
    size_before = len(df)
    df = df[df[UNTRANSFORMED_TARGET].notna() & (df[UNTRANSFORMED_TARGET] > 0)]
    logger.info("Dropped %s rows with missing/invalid target", size_before - len(df))
    
    # -- Drop rows with negative mileage/year
    if "Mileage" in df.columns:
        df = df[(df["Mileage"].isna()) | (df["Mileage"] >= 0)] # keep rows with missing mileage, but drop negative mileage
    if "Year" in df.columns:
        df = df[(df["Year"].isna()) | ((df["Year"] >= 1950) & (df["Year"] <= CURRENT_YEAR + 1))] # keep rows with missing year, but drop implausible years

    # -- Normalize categorical text fields: trim whitespace, standard casing
    categorical_text_cols = CATEGORICAL_FEATURES
    for col in categorical_text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            
    # -- Reset index
    df = df.reset_index(drop=True)
    return df


def load_and_clean(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load -> validate -> clean.
    
    Parameters
    ----------
    path : str | Path
        Location of the CSV file. Defaults to data/automotive_dataset.csv.
        
    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.
    """
    df = load_raw_data(path)
    validate_schema(df)
    df = clean_data(df)
    return df

# just a test run to check if loader.py is loaded correctly
if __name__ == "__main__":
    df = load_and_clean()
    print(f"Loaded and cleaned data: {len(df)} rows, {df.shape[1]} columns")