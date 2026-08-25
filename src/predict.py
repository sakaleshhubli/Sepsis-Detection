"""
Inference script for the Sepsis Detection project.
Loads the saved XGBoost model + feature pipeline and produces
per-timestep sepsis probability predictions for a patient's ICU stay.
"""

import os
import pickle

import numpy as np
import pandas as pd

from data_loader import load_config
from features import engineer_features


def load_model_bundle(config: dict) -> dict:
    """Load the trained model, fill_values, and feature_cols saved by train.py."""
    model_path = os.path.join(config["paths"]["model_dir"], config["paths"]["model_name"])
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def predict(
    raw_df: pd.DataFrame,
    config: dict,
    bundle: dict,
    threshold: float = 0.475,
) -> pd.DataFrame:
    """
    Run the full feature pipeline + model on a patient's raw hourly readings
    and return per-timestep sepsis probability and prediction.

    raw_df must contain the same raw columns used in training (vitals, labs,
    static fields, Patient_ID, ICULOS) — SepsisLabel is not required.

    Returns a DataFrame with columns: Patient_ID, ICULOS, sepsis_probability,
    sepsis_prediction (at the given threshold).
    """
    model = bundle["model"]
    fill_values = bundle["fill_values"]
    feature_cols = bundle["feature_cols"]

    id_col = config["columns"]["id_col"]
    time_col = config["columns"]["time_col"]

    feat_df, _ = engineer_features(raw_df, config, fill_values=fill_values)

    proba = model.predict_proba(feat_df[feature_cols])[:, 1]

    result = feat_df[[id_col, time_col]].copy()
    result["sepsis_probability"] = proba
    result["sepsis_prediction"] = (proba >= threshold).astype(int)

    return result


if __name__ == "__main__":
    # Quick smoke test: predict on a few patients from the test set
    config = load_config()
    bundle = load_model_bundle(config)

    test_path = os.path.join(config["data"]["processed_dir"], "test.parquet")
    test_df = pd.read_parquet(test_path)

    id_col = config["columns"]["id_col"]
    sample_patients = test_df[id_col].unique()[:3]
    sample_df = test_df[test_df[id_col].isin(sample_patients)].copy()

    preds = predict(sample_df, config, bundle)
    print(preds.to_string(index=False))

    # Compare against ground truth for sanity check
    actual = sample_df[[id_col, config["columns"]["time_col"], config["columns"]["label_col"]]]
    merged = preds.merge(actual, on=[id_col, config["columns"]["time_col"]])
    print(f"\nActual sepsis rate in sample: {merged[config['columns']['label_col']].mean():.4f}")
    print(f"Predicted sepsis rate in sample: {merged['sepsis_prediction'].mean():.4f}")