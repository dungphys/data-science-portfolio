"""
Inference helper: apply the same feature engineering used at training
time, then predict resale price for new/raw vehicle records.
"""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.build_features import build_features
from src.models.train import MODEL_FEATURES


def predict_price(pipeline: Pipeline, raw_df: pd.DataFrame) -> pd.Series:
    """Predict Selling_Price for raw (non-feature-engineered) vehicle rows.

    Parameters
    ----------
    pipeline : Pipeline
        fitted sklearn Pipeline (preprocessor + regressor)
    raw_df : pd.DataFrame
        DataFrame with the original raw columns (Make, Model, Year, ...)
        Does NOT need to already contain engineered columns.
    """
    engineered = build_features(raw_df)
    missing = [c for c in MODEL_FEATURES if c not in engineered.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    X = engineered[MODEL_FEATURES]
    preds = pipeline.predict(X)
    return pd.Series(preds, index=raw_df.index, name="Predicted_Selling_Price")
