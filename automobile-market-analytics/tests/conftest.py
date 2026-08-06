"""
Shared pytest fixtures.

Generate a small, deterministic synthetic dataset that matches the
real schema (Make, Model, Year, ... Selling_Price) rather than
depending on the real data/automotive_dataset.csv file. This keeps
tests fast, deterministic, and independent of the (potentially large,
not-checked-in) production dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_sample_df() -> pd.DataFrame:
    """Generate a small raw sample

    Returns:
        pd.DataFrame
    """
    rng = np.random.default_rng(42)
    n = 200

    makes = ["Toyota", "Honda", "BMW", "Mercedes-Benz", "Audi"]
    models = ["Camry", "Civic", "3 Series", "C-Class", "A4"]
    fuel_types = ["Petrol", "Diesel", "Hybrid", "Electric"]
    transmissions = ["Manual", "Automatic"]
    service_hist = ["Full Service", "Partial Service", "No Service"]
    colors = ["Black", "White", "Silver", "Blue", "Red"]
    body_types = ["Sedan", "SUV", "Truck", "Coupe", "Hatchback"]
    drivetrains = ["FWD", "AWD", "RWD", "4WD"]
    states = ["CA", "TX", "NY", "FL", "WA"]

    idx = rng.integers(0, len(makes), n)
    year = rng.integers(2005, 2025, n)
    mileage = rng.integers(500, 150000, n).astype(float)
    horsepower = rng.integers(90, 500, n).astype(float)
    engine_size = np.round(rng.uniform(1.0, 5.0, n), 1)
    torque = rng.integers(100, 500, n).astype(float)
    owners = rng.integers(1, 5, n)
    accident = rng.integers(0, 2, n)
    fuel_eff = np.round(rng.uniform(15, 55, n), 1)

    # Construct a target with a known-ish relationship so correlation
    # tests have something real to detect.
    base_price = 40000
    price = (
        base_price
        - (2026 - year) * 900
        - mileage * 0.05
        + horsepower * 30
        - accident * 2500
        + rng.normal(0, 1500, n)
    )
    price = np.clip(price, 1000, None)

    df = pd.DataFrame(
        {
            "Make": [makes[i] for i in idx],
            "Model": [models[i] for i in idx],
            "Year": year,
            "Fuel_Type": rng.choice(fuel_types, n),
            "Transmission": rng.choice(transmissions, n),
            "Engine_Size": engine_size,
            "Mileage": mileage,
            "Horsepower": horsepower,
            "Torque": torque,
            "Owners": owners,
            "Accident_History": accident,
            "Service_History": rng.choice(service_hist, n),
            "Color": rng.choice(colors, n),
            "Body_Type": rng.choice(body_types, n),
            "Drivetrain": rng.choice(drivetrains, n),
            "Fuel_Efficiency": fuel_eff,
            "Location": rng.choice(states, n),
            "Selling_Price": price,
        }
    )
    return df


@pytest.fixture
def sample_csv_path(tmp_path, raw_sample_df):
    path = tmp_path / "automotive_dataset.csv"
    raw_sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def cleaned_df(raw_sample_df):
    from src.data.loader import clean_data

    return clean_data(raw_sample_df)


@pytest.fixture
def featured_df(cleaned_df):
    from src.features.build_features import build_features

    return build_features(cleaned_df)