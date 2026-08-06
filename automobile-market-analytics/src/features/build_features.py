"""
Feature engineering for the automotive resale-price model.

Add derived features that are known (from domain knowledge) to be
predictive of resale price but aren't directly present as raw columns:

- Vehicle_Age: how old the car is (current year - manufacturing year).
- Mileage_Per_Year: average annual usage (mileage / vehicle age). This is a proxy for wear-and-tear.
- Power_Per_Liter: horsepower per liter of displacement, a proxy for engine performance/efficiency tier.
- Selling_Price_Log: log-transformed selling price, used for modeling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CURRENT_YEAR

def add_vehicle_age(df: pd.DataFrame) -> pd.DataFrame:
    """Add a new column 'Vehicle_Age' to the DataFrame.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Year' column representing the manufacturing year of the vehicle.
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with an additional 'Vehicle_Age' column
    """
    df = df.copy()
    df["Vehicle_Age"] = (CURRENT_YEAR - df["Year"]).clip(lower=0)
    return df


def add_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add a new column 'Mileage_Per_Year' to the DataFrame.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Mileage' column representing the total mileage of the vehicle and a 'Vehicle_Age' column.
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with an additional 'Mileage_Per_Year' column
    """
    df = df.copy()
    # Avoid division by zero for brand-new vehicles (age 0 -> treat as age 0.5 for calculation purposes)
    safe_age = df["Vehicle_Age"].replace(0, 0.5)
    df["Mileage_Per_Year"] = df["Mileage"] / safe_age
    return df


def add_power_per_liter(df: pd.DataFrame) -> pd.DataFrame:
    """Add a new column 'Power_Per_Liter' to the DataFrame.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Engine_Size' column representing the engine size of the vehicle and a 'Horsepower' column.
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with an additional 'Power_Per_Liter' column
    """
    df = df.copy()
    safe_engine = df["Engine_Size"].replace(0, np.nan)  # Avoid division by zero for vehicles with missing engine size (0 -> treat as NaN for calculation purposes)
    df["Power_Per_Liter"] = df["Horsepower"] / safe_engine 
    return df

def add_log_transformed_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a new column 'Selling_Price_Log' to the DataFrame.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'Selling_Price' column.
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with an additional 'Selling_Price_Log' column
    """
    df = df.copy()
    df["Selling_Price_Log"] = np.log(df["Selling_Price"]) 
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in the correct order.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw features.
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with engineered features added.
    """
    df = add_vehicle_age(df)
    df = add_mileage_per_year(df)
    df = add_power_per_liter(df)
    df = add_log_transformed_target(df)
    return df

#* just a test run to check if build_features.py is loaded correctly
if __name__ == "__main__":
    df = pd.DataFrame({
        "Year": [2020, 2015, 2010, 2007],
        "Mileage": [10000, 50000, 80000, 120000],
        "Engine_Size": [2.0, 1.5, 0.0, 2.5],
        "Horsepower": [150, 120, 250, 180],
        "Selling_Price": [20000, 15000, 10000, 8000]
    })
    df = build_features(df)
    print(df)
