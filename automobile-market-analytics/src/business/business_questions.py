"""
Business-question analysis functions.

Each function answers ONE concrete business question using the cleaned
+ feature-engineered dataset (and, where relevant, the trained model's
feature importances). Every function returns a plain pandas object
(DataFrame/Series/dict) so it is trivially unit-testable and reusable
outside of the report script.

See reports/business_report.md for the narrative writeup of results,
and README.md for the methodology summary.
"""
from __future__ import annotations

import pandas as pd

from src.config import UNTRANSFORMED_TARGET, TRANSFORMED_TARGET


def q1_top_price_drivers(feature_importance_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Q1: Which vehicle attributes most influence resale price?

    Methodology: extract feature_importances_ from the trained
    RandomForestRegressor (mean decrease in impurity across trees),
    mapped back to human-readable (one-hot-expanded) feature names.
    
    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        DataFrame containing feature names and their importance values.
    top_n : int, default=10
        The number of top features to return.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the top_n features and their importance values, sorted descending.
    """
    return feature_importance_df.head(top_n)


def q2_accident_price_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """Q2: How much does a prior accident reduce resale value?

    Methodology: group by Accident_History, compare mean/median price,
    and express the gap as a percentage of the no-accident mean
    (a simple, interpretable "penalty" figure for pricing decisions).
    """
    grouped = df.groupby("Accident_History")[UNTRANSFORMED_TARGET].agg(["mean", "median", "count"])
    grouped.index = grouped.index.map({0: "No Accident", 1: "Accident"})
    if "No Accident" in grouped.index and "Accident" in grouped.index:
        no_acc_mean = grouped.loc["No Accident", "mean"]
        acc_mean = grouped.loc["Accident", "mean"]
        penalty_pct = (no_acc_mean - acc_mean) / no_acc_mean * 100
        grouped["penalty_vs_no_accident_pct"] = [
            0.0 if idx == "No Accident" else round(penalty_pct, 2) for idx in grouped.index
        ]
    return grouped


def q3_depreciation_by_age(df: pd.DataFrame, bins: list[int] | None = None) -> pd.DataFrame:
    """Q3: What does the depreciation curve look like as vehicles age?

    Methodology: bucket Vehicle_Age into ranges and compute median
    selling price per bucket, plus the percentage drop from the
    newest bucket -- this approximates a depreciation curve without
    needing original MSRP data.
    """
    if bins is None:
        bins = [-1, 1, 3, 5, 8, 12, 100]
    labels = ["0-1 yr", "2-3 yr", "4-5 yr", "6-8 yr", "9-12 yr", "13+ yr"]
    age_bucket = pd.cut(df["Vehicle_Age"], bins=bins, labels=labels)
    grouped = df.groupby(age_bucket, observed=True)[UNTRANSFORMED_TARGET].agg(["median", "mean", "count"])
    baseline = grouped["median"].iloc[0]
    grouped["pct_of_newest_bucket"] = (grouped["median"] / baseline * 100).round(1)
    return grouped


def q4_price_by_fuel_and_body(df: pd.DataFrame) -> pd.DataFrame:
    """Q4: Which fuel type / body type combinations command the highest resale value?

    Methodology: pivot table of median Selling_Price by Fuel_Type x
    Body_Type. Median is used (not mean) to reduce sensitivity to
    high-end outliers (e.g. rare luxury trims).
    """
    pivot = df.pivot_table(
        index="Body_Type", columns="Fuel_Type", values=UNTRANSFORMED_TARGET, aggfunc="median"
    )
    return pivot.round(0)


def q5_service_history_value(df: pd.DataFrame) -> pd.DataFrame:
    """Q5: Does a fuller service history translate into measurable resale value?

    Methodology: group by Service_History category, compare median
    price, and compute the dollar and percentage premium of "Full
    Service" over "No Service" -- directly useful for setting
    maintenance-record-based pricing adjustments.
    """
    grouped = df.groupby("Service_History")[UNTRANSFORMED_TARGET].agg(["mean", "median", "count"])
    grouped = grouped.reindex(["No Service", "Partial Service", "Full Service"]).dropna(how="all")
    if "No Service" in grouped.index and "Full Service" in grouped.index:
        no_service = grouped.loc["No Service", "median"]
        full_service = grouped.loc["Full Service", "median"]
        premium_pct = (full_service - no_service) / no_service * 100
        grouped["premium_vs_no_service_pct"] = round(premium_pct, 2)
    return grouped


def q6_regional_price_variation(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Q6: Which states/regions show the highest and lowest average resale prices?

    Methodology: group by Location (state), compute mean price and
    sample count; filter out states with very few listings (< 5) to
    avoid noisy small-sample averages skewing conclusions.
    """
    grouped = df.groupby("Location")[UNTRANSFORMED_TARGET].agg(["mean", "median", "count"])
    grouped = grouped[grouped["count"] >= 5].sort_values("mean", ascending=False)
    return grouped.head(top_n)


def q7_mileage_efficiency_relationship(df: pd.DataFrame) -> pd.Series:
    """Q7: Does higher mileage-adjusted usage (Mileage_Per_Year) correlate with lower price?

    Methodology: Pearson correlation between Mileage_Per_Year and
    Selling_Price, controlling implicitly for age since the metric is
    already annualized. A strong negative correlation supports pricing
    "heavily used" vehicles at a discount beyond simple age-based
    depreciation.
    """
    corr = df[["Mileage_Per_Year", UNTRANSFORMED_TARGET]].corr().loc["Mileage_Per_Year", UNTRANSFORMED_TARGET]
    return pd.Series({"correlation_mileage_per_year_vs_price": round(float(corr), 4)})


def run_all_business_questions(df: pd.DataFrame, feature_importance_df: pd.DataFrame) -> dict:
    """Run every business question function and collect results in a dict."""
    return {
        "q1_top_price_drivers": q1_top_price_drivers(feature_importance_df),
        "q2_accident_price_penalty": q2_accident_price_penalty(df),
        "q3_depreciation_by_age": q3_depreciation_by_age(df),
        "q4_price_by_fuel_and_body": q4_price_by_fuel_and_body(df),
        "q5_service_history_value": q5_service_history_value(df),
        "q6_regional_price_variation": q6_regional_price_variation(df),
        "q7_mileage_efficiency_relationship": q7_mileage_efficiency_relationship(df),
    }
