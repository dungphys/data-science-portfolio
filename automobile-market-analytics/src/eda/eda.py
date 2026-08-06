"""
Exploratory Data Analysis utilities.

Two kinds of functions:
1. Pure, testable summary functions that return DataFrames/dicts
(no plotting, no I/O) -> easy to unit test.
2. Plotting functions that save figures to results/figures/ -> exercised
via the run_eda.py script, smoke-tested (not pixel-tested) in pytest.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, required for Docker/CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    FIGURES_DIR, 
    NUMERIC_FEATURES, 
    UNTRANSFORMED_TARGET, 
    CATEGORICAL_FEATURES, 
    BINARY_FEATURES,
    
)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Pure summary functions
# ---------------------------------------------------------------------------
def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, sorted descending.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'missing_count' and 'missing_pct', 
        sorted by 'missing_count' in descending order.
    """
    counts = df.isna().sum()
    pct = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": counts, "missing_pct": pct})
    return summary.sort_values("missing_count", ascending=False)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with descriptive statistics
    """
    cols = [c for c in NUMERIC_FEATURES + [UNTRANSFORMED_TARGET] if c in df.columns]
    return df[cols].describe().T


def correlation_with_target(df: pd.DataFrame, target: str = UNTRANSFORMED_TARGET) -> pd.Series:
    """Pearson correlation of numeric columns with the target, sorted.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    target : str, optional
        Target column name, by default UNTRANSFORMED_TARGET
        
    Returns
    -------
    pd.Series
        Series of correlations, sorted by absolute value descending.
    """
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[cols + [target]].corr(numeric_only=True)[target].drop(target)
    return corr.sort_values(key=abs, ascending=False)


def price_by_category(df: pd.DataFrame, category_col: str, target: str = UNTRANSFORMED_TARGET) -> pd.DataFrame:
    """Mean/median/count of target grouped by a categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    category_col : str
        Categorical column to group by.
    target : str, optional
        Target column name, by default UNTRANSFORMED_TARGET
    
    Returns
    -------
    pd.DataFrame
        DataFrame with mean, median, and count of target per category.
    """
    grouped = (
        df.groupby(category_col)[target]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
    )
    return grouped


def outlier_bounds_iqr(series: pd.Series, k: float) -> tuple[float, float]:
    """Return (lower, upper) Tukey IQR bounds for outlier detection.
    
    Parameters
    ----------
    series : pd.Series
        Input series to analyze.
    k : float
        Multiplier for the IQR to determine the bounds. 
        Common values: 1.5 (mild outliers), 3.0 (extreme outliers).
        
    Returns
    -------
    tuple[float, float]
        Lower and upper bounds for outlier detection.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


# ---------------------------------------------------------------------------
# Plotting functions (save to disk, headless)
# ---------------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Plot histogram of the target variable and save to disk.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column.
    out_dir : Path, optional
        Directory to save the figure, by default FIGURES_DIR
    
    Returns
    -------
    Path
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[UNTRANSFORMED_TARGET].dropna(), kde=True, ax=ax, color="#2b6cb0")
    ax.set_title("Distribution of Selling Price")
    ax.set_xlabel("Selling Price ($)")
    out_path = Path(out_dir) / "target_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Plot correlation heatmap of numeric features and save to disk.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    out_dir : Path, optional
        Directory to save the figure, by default FIGURES_DIR

    Returns
    -------
    Path
        Path to the saved figure.
    """
    cols = [c for c in NUMERIC_FEATURES + [UNTRANSFORMED_TARGET] if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    out_path = Path(out_dir) / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_scatter_relationship(
    df: pd.DataFrame,
    x: str,
    y: str = UNTRANSFORMED_TARGET,
    hue: str | None = None,
    title: str | None = None,
    filename: str | None = None,
    out_dir: Path = FIGURES_DIR,
    alpha: float = 0.6,
) -> Path:
    """Generic scatter plot of `y` vs `x`, optionally colored by `hue`.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    x : str
        Column name for the x-axis.
    y : str, optional
        Column name for the y-axis, by default UNTRANSFORMED_TARGET
    hue : str | None, optional
        Column name for color grouping, by default None
    title : str | None, optional
        Title of the plot, by default None (auto-generated)
    filename : str | None, optional
        Filename for the saved figure, by default None (auto-generated)
    out_dir : Path, optional
        Directory to save the figure, by default FIGURES_DIR
    alpha : float, optional
        Transparency level for scatter points, by default 0.6
    
    Returns
    -------
    Path
        Path to the saved figure.  
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=alpha, ax=ax)
    ax.set_title(title or f"{y} vs {x}")
    out_path = Path(out_dir) / (filename or f"{y.lower()}_vs_{x.lower()}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

def plot_boxplot_by_category(
    df: pd.DataFrame,
    category_col: str,
    target: str = UNTRANSFORMED_TARGET,
    title: str | None = None,
    filename: str | None = None,
    out_dir: Path = FIGURES_DIR,
    order_by: str = "median",
    ascending: bool = False,
    rotation: int = 0,
) -> Path:
    """Generic boxplot of `target` grouped by any categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    category_col : str
        Categorical column to group by.
    target : str, optional
        Target column name, by default UNTRANSFORMED_TARGET
    title : str | None, optional
        Title of the plot, by default None (auto-generated)
    filename : str | None, optional
        Filename for the saved figure, by default None (auto-generated)
    out_dir : Path, optional
        Directory to save the figure, by default FIGURES_DIR
    order_by : which statistic to sort categories by ("median" or "mean").
    ascending : bool, optional
        Whether to sort categories in ascending order, by default False
    rotation : int, optional
        Rotation angle for x-axis tick labels, by default 0    
    
    Returns
    -------
    Path
        Path to the saved figure.  
    """
    order = (
        df.groupby(category_col)[target].agg(order_by).sort_values(ascending=ascending).index
    )
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df, x=category_col, y=target, order=order, ax=ax)
    ax.set_title(title or f"{target} by {category_col}")
    if rotation:
        ax.tick_params(axis="x", rotation=rotation)
    out_path = Path(out_dir) / (filename or f"{target.lower()}_by_{category_col.lower()}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

def plot_mid_target_vs_category(
    df: pd.DataFrame,
    category_col: str,
    target: str = UNTRANSFORMED_TARGET,
    title: str | None = None, 
    filename: str | None = None,
    out_dir: Path = FIGURES_DIR,
    rotation: int = 0, 
) -> Path:
    """Line plot of mean/median target grouped by a categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    category_col : str
        Categorical column to group by.
    target : str, optional
        Target column name, by default UNTRANSFORMED_TARGET
    title : str | None, optional
        Title of the plot, by default None (auto-generated)
    filename : str | None, optional
        Filename for the saved figure, by default None (auto-generated)
    out_dir : Path, optional
        Directory to save the figure, by default FIGURES_DIR
    rotation : int, optional
        Rotation angle for x-axis tick labels, by default 0    
    
    Returns
    -------
    Path
        Path to the saved figure.  
    """
    median_target = df.groupby(category_col)[target].median().sort_values(ascending=False)
    mean_target = df.groupby(category_col)[target].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(median_target.index.astype(str), median_target.values, 
            '-o', color="red", label = "Median Price")
    ax.bar(mean_target.index.astype(str), mean_target.values, 
            color="blue", label = "Mean Price", alpha=0.65)
    ax.set_title(title or f"Mean and Median of {target} by {category_col}")
    ax.set_ylabel(f"{target}")
    ax.set_xlabel(f"{category_col}")
    ax.legend(loc="best")
    if rotation:
        ax.tick_params(axis="x", rotation=rotation)
    out_path = Path(out_dir) / (filename or f"mean_median_{target.lower()}_by_{category_col.lower()}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

def generate_all_plots(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate all EDA plots and save to disk.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    out_dir : Path, optional
        Directory to save the figures, by default FIGURES_DIR
    
    Returns
    -------
    list[Path]
        List of paths to the saved figures.
    """
    paths = []
    paths.append(plot_target_distribution(df, out_dir))
    paths.append(plot_correlation_heatmap(df, out_dir))
    
    # scatter plots for numeric features vs target
    for num_col in NUMERIC_FEATURES:
        if num_col in df.columns and num_col != "Year":  # Year is not a continuous variable, so skip scatter plot
            paths.append(plot_scatter_relationship(df, x=num_col, y=UNTRANSFORMED_TARGET, out_dir=out_dir))
    paths.append(plot_scatter_relationship(df, 
                                        x="Fuel_Efficiency", 
                                        y=UNTRANSFORMED_TARGET, hue="Fuel_Type", 
                                        out_dir=out_dir, filename="selling_price_vs_fuel.png"))
    paths.append(plot_scatter_relationship(df, 
                                        x="Engine_Size", 
                                        y=UNTRANSFORMED_TARGET, hue="Fuel_Type", 
                                        out_dir=out_dir, filename="selling_price_vs_engine_fuel.png"))
    # boxplots for categorical features vs target
    for cat_col in CATEGORICAL_FEATURES + BINARY_FEATURES + ["Vehicle_Age"]:
        if cat_col in df.columns:
            paths.append(plot_boxplot_by_category(df, category_col=cat_col, target=UNTRANSFORMED_TARGET, out_dir=out_dir))
    
    cat_to_plot_mid = ["Body_Type", "Owners", "Make", "Vehicle_Age"]
    rotation = [0, 0, 60, 0]
    for cat, rot in zip(cat_to_plot_mid,rotation):       
        paths.append(plot_mid_target_vs_category(df, category_col=cat, target=UNTRANSFORMED_TARGET, out_dir=out_dir, rotation=rot))    
    return paths