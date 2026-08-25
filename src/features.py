"""
Feature engineering for the Sepsis Detection project.
Builds missingness indicators, per-patient forward-fill, and rolling
window statistics on top of the raw vitals/labs.
"""

import pandas as pd
import numpy as np


def add_missingness_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add a {col}_missing binary flag for each column, before any imputation."""
    df = df.copy()
    for col in cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def forward_fill_by_patient(
    df: pd.DataFrame, cols: list[str], id_col: str, time_col: str
) -> pd.DataFrame:
    """Forward-fill vitals/labs within each patient's timeline, sorted by time."""
    df = df.sort_values([id_col, time_col]).copy()
    df[cols] = df.groupby(id_col)[cols].ffill()
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    id_col: str,
    time_col: str,
    window: int = 6,
) -> pd.DataFrame:
    """
    Add rolling mean/std/min/max per column over a window of `window` hours,
    computed within each patient's timeline (after sorting by time).
    """
    df = df.sort_values([id_col, time_col]).copy()
    grouped = df.groupby(id_col)[cols]

    roll_mean = grouped.rolling(window=window, min_periods=1).mean().reset_index(drop=True)
    roll_std = grouped.rolling(window=window, min_periods=1).std().reset_index(drop=True)
    roll_min = grouped.rolling(window=window, min_periods=1).min().reset_index(drop=True)
    roll_max = grouped.rolling(window=window, min_periods=1).max().reset_index(drop=True)

    for col in cols:
        df[f"{col}_roll_mean"] = roll_mean[col].values
        df[f"{col}_roll_std"] = roll_std[col].values
        df[f"{col}_roll_min"] = roll_min[col].values
        df[f"{col}_roll_max"] = roll_max[col].values

    # rolling std is NaN when only 1 point is in the window — fill with 0 (no variability observed yet)
    std_cols = [f"{col}_roll_std" for col in cols]
    df[std_cols] = df[std_cols].fillna(0)

    return df

def add_clinical_features(
    df: pd.DataFrame, id_col: str, time_col: str
) -> pd.DataFrame:
    """
    Add domain-informed clinical features on top of forward-filled vitals/labs:
    Shock Index, SIRS criteria count, rate-of-change deltas, and BUN/Creatinine ratio.
    Must be called AFTER forward_fill_by_patient so values are populated.
    """
    df = df.sort_values([id_col, time_col]).copy()

    # Shock Index = HR / SBP (elevated = hemodynamic instability)
    df["shock_index"] = df["HR"] / df["SBP"].replace(0, np.nan)
    df["shock_index"] = df["shock_index"].replace([np.inf, -np.inf], np.nan)

    # SIRS criteria count (0-4): Temp, HR, Resp, WBC thresholds
    sirs_temp = ((df["Temp"] > 38) | (df["Temp"] < 36)).astype(int)
    sirs_hr = (df["HR"] > 90).astype(int)
    sirs_resp = (df["Resp"] > 20).astype(int)
    sirs_wbc = ((df["WBC"] > 12) | (df["WBC"] < 4)).astype(int)
    df["sirs_count"] = sirs_temp + sirs_hr + sirs_resp + sirs_wbc

    # Rate-of-change: delta from previous reading, within each patient
    delta_cols = ["HR", "SBP", "MAP", "Resp", "O2Sat", "Temp"]
    for col in delta_cols:
        df[f"{col}_delta"] = df.groupby(id_col)[col].diff().fillna(0)

    # BUN/Creatinine ratio (renal hypoperfusion indicator)
    df["bun_creatinine_ratio"] = df["BUN"] / df["Creatinine"].replace(0, np.nan)
    df["bun_creatinine_ratio"] = df["bun_creatinine_ratio"].replace([np.inf, -np.inf], np.nan)

    return df


def fill_remaining_na(
    df: pd.DataFrame, cols: list[str], fill_values: pd.Series = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fill any remaining NaNs (e.g. leading NaNs before a patient's first
    reading) with provided fill_values (medians), or compute them from df
    if not provided. Always compute fill_values from TRAIN only and reuse
    for val/test to avoid leakage.
    """
    df = df.copy()
    if fill_values is None:
        fill_values = df[cols].median()
    df[cols] = df[cols].fillna(fill_values)
    return df, fill_values


def engineer_features(
    df: pd.DataFrame,
    config: dict,
    fill_values: pd.Series = None,
    window: int = 6,
) -> tuple[pd.DataFrame, pd.Series]:
    id_col = config["columns"]["id_col"]
    time_col = config["columns"]["time_col"]
    vital_cols = config["columns"]["vital_cols"]
    lab_cols = config["columns"]["lab_cols"]
    measured_cols = vital_cols + lab_cols

    df = add_missingness_indicators(df, measured_cols)
    df = forward_fill_by_patient(df, measured_cols, id_col, time_col)
    df = add_clinical_features(df, id_col, time_col)          # <-- new
    df = add_rolling_features(df, vital_cols, id_col, time_col, window=window)

    rolling_cols = [
        f"{c}_roll_{stat}" for c in vital_cols for stat in ["mean", "std", "min", "max"]
    ]
    clinical_cols = [
        "shock_index", "sirs_count", "bun_creatinine_ratio",
        "HR_delta", "SBP_delta", "MAP_delta", "Resp_delta", "O2Sat_delta", "Temp_delta",
    ]
    all_numeric_cols = measured_cols + rolling_cols + clinical_cols   # <-- updated

    df, fill_values = fill_remaining_na(df, all_numeric_cols, fill_values)

    return df, fill_values


if __name__ == "__main__":
    from data_loader import load_and_validate

    train_df, val_df, test_df, config = load_and_validate()

    print("\nEngineering features on train...")
    train_feat, fill_values = engineer_features(train_df, config)
    print(f"Train shape after feature engineering: {train_feat.shape}")

    print("Engineering features on val/test (using train fill values)...")
    val_feat, _ = engineer_features(val_df, config, fill_values=fill_values)
    test_feat, _ = engineer_features(test_df, config, fill_values=fill_values)

    print(f"Val shape: {val_feat.shape}, Test shape: {test_feat.shape}")
    print(f"\nRemaining NaNs in train: {train_feat.isna().sum().sum()}")