import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    SchemaError, 
    clean_data, 
    load_and_clean, 
    load_raw_data, 
    validate_schema
)


def test_load_raw_data_reads_csv(sample_csv_path, raw_sample_df):
    df = load_raw_data(sample_csv_path)
    assert len(df) == len(raw_sample_df)
    assert set(df.columns) == set(raw_sample_df.columns)


def test_load_raw_data_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_path)


def test_validate_schema_passes_on_full_schema(raw_sample_df):
    # Should not raise
    validate_schema(raw_sample_df)


def test_validate_schema_raises_on_missing_columns(raw_sample_df):
    broken = raw_sample_df.drop(columns=["Selling_Price"])
    with pytest.raises(SchemaError):
        validate_schema(broken)


def test_clean_data_drops_duplicates(raw_sample_df):
    with_dupes = pd.concat([raw_sample_df, raw_sample_df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(with_dupes)
    assert len(cleaned) == len(raw_sample_df)


def test_clean_data_drops_invalid_target_rows(raw_sample_df):
    df = raw_sample_df.copy()
    df.loc[0, "Selling_Price"] = np.nan
    df.loc[1, "Selling_Price"] = -100
    cleaned = clean_data(df)
    assert cleaned["Selling_Price"].notna().all()
    assert (cleaned["Selling_Price"] > 0).all()
    assert len(cleaned) == len(df) - 2


def test_clean_data_coerces_numeric_columns(raw_sample_df):
    df = raw_sample_df.copy()
    df["Mileage"] = df["Mileage"].astype(str)
    cleaned = clean_data(df)
    assert pd.api.types.is_numeric_dtype(cleaned["Mileage"])


def test_clean_data_strips_whitespace_in_categoricals(raw_sample_df):
    df = raw_sample_df.copy()
    df.loc[0, "Make"] = "  Toyota  "
    cleaned = clean_data(df)
    assert cleaned.loc[0, "Make"] == "Toyota"


def test_clean_data_filters_invalid_year(raw_sample_df):
    df = raw_sample_df.copy()
    df.loc[0, "Year"] = 3000
    cleaned = clean_data(df)
    assert (cleaned["Year"].dropna() <= 2100).all()


def test_load_and_clean_end_to_end(sample_csv_path):
    df = load_and_clean(sample_csv_path)
    assert len(df) > 0
    assert df["Selling_Price"].notna().all()