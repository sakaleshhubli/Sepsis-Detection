import pandas as pd
import numpy as np


def add_missingness_indicators(df, cols):
    df = df.copy()
    for col in cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def forward_fill_by_patient(df, cols, id_col, time_col):
    df = df.sort_values([id_col, time_col]).copy()
    df[cols] = df.groupby(id_col)[cols].ffill()
    return df


def add_rolling_features(df, cols, id_col, time_col, window=6):
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
    std_cols = [f"{col}_roll_std" for col in cols]
    df[std_cols] = df[std_cols].fillna(0)
    return df


def add_clinical_features(df, id_col, time_col):
    df = df.sort_values([id_col, time_col]).copy()
    df["shock_index"] = df["HR"] / df["SBP"].replace(0, np.nan)
    df["shock_index"] = df["shock_index"].replace([np.inf, -np.inf], np.nan)
    sirs_temp = ((df["Temp"] > 38) | (df["Temp"] < 36)).astype(int)
    sirs_hr = (df["HR"] > 90).astype(int)
    sirs_resp = (df["Resp"] > 20).astype(int)
    sirs_wbc = ((df["WBC"] > 12) | (df["WBC"] < 4)).astype(int)
    df["sirs_count"] = sirs_temp + sirs_hr + sirs_resp + sirs_wbc
    delta_cols = ["HR", "SBP", "MAP", "Resp", "O2Sat", "Temp"]
    for col in delta_cols:
        df[f"{col}_delta"] = df.groupby(id_col)[col].diff().fillna(0)
    df["bun_creatinine_ratio"] = df["BUN"] / df["Creatinine"].replace(0, np.nan)
    df["bun_creatinine_ratio"] = df["bun_creatinine_ratio"].replace([np.inf, -np.inf], np.nan)
    return df


def fill_remaining_na(df, cols, fill_values=None):
    df = df.copy()
    if fill_values is None:
        fill_values = df[cols].median()
    df[cols] = df[cols].fillna(fill_values)
    return df, fill_values


def engineer_features(df, config, fill_values=None, window=6):
    id_col = config["columns"]["id_col"]
    time_col = config["columns"]["time_col"]
    vital_cols = config["columns"]["vital_cols"]
    lab_cols = config["columns"]["lab_cols"]
    measured_cols = vital_cols + lab_cols

    df = add_missingness_indicators(df, measured_cols)
    df = forward_fill_by_patient(df, measured_cols, id_col, time_col)
    df = add_clinical_features(df, id_col, time_col)
    df = add_rolling_features(df, vital_cols, id_col, time_col, window=window)

    rolling_cols = [f"{c}_roll_{stat}" for c in vital_cols for stat in ["mean", "std", "min", "max"]]
    clinical_cols = [
        "shock_index", "sirs_count", "bun_creatinine_ratio",
        "HR_delta", "SBP_delta", "MAP_delta", "Resp_delta", "O2Sat_delta", "Temp_delta",
    ]
    static_cols = config["columns"]["static_cols"]
    all_numeric_cols = measured_cols + rolling_cols + clinical_cols + static_cols

    df, fill_values = fill_remaining_na(df, all_numeric_cols, fill_values)

    return df, fill_values