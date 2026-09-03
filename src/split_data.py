"""
One-time script: splits the raw PhysioNet Dataset.csv into
train/val/test CSVs by PATIENT (not by row), stratified on whether
the patient is ever septic, so the ~7% septic-patient rate stays
consistent across splits and no patient's hours leak across splits.

Run: python src/split_data.py
Reads:  data/raw/Dataset.csv          (path from config.yaml)
Writes: data/processed/train.csv
        data/processed/val.csv
        data/processed/test.csv
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_config


def split_patients(patient_ids: np.ndarray, ever_septic: np.ndarray, seed: int):
    """70/15/15 patient-level split, stratified on ever_septic."""
    train_ids, temp_ids, train_strat, temp_strat = train_test_split(
        patient_ids, ever_septic, test_size=0.30, stratify=ever_septic, random_state=seed,
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, stratify=temp_strat, random_state=seed,
    )
    return set(train_ids), set(val_ids), set(test_ids)


def main(seed: int = 42):
    config = load_config()
    id_col = config["columns"]["id_col"]
    label_col = config["columns"]["label_col"]
    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]

    print(f"Loading {raw_path} ...")
    df = pd.read_csv(raw_path, index_col=0)
    # Guard against a leading unnamed index column from how the file was exported
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    patient_labels = df.groupby(id_col)[label_col].max()  # 1 if ever septic
    patient_ids = patient_labels.index.values
    ever_septic = patient_labels.values

    print(f"Total patients: {len(patient_ids)} | ever-septic rate: {ever_septic.mean():.4f}")

    train_ids, val_ids, test_ids = split_patients(patient_ids, ever_septic, seed)

    train_df = df[df[id_col].isin(train_ids)]
    val_df = df[df[id_col].isin(val_ids)]
    test_df = df[df[id_col].isin(test_ids)]

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_patients = split_df[id_col].nunique()
        septic_rate = split_df.groupby(id_col)[label_col].max().mean()
        row_rate = split_df[label_col].mean()
        print(f"{name:5s}: {len(split_df):>8,} rows | {n_patients:>6,} patients "
              f"| ever-septic rate: {septic_rate:.4f} | row-level sepsis rate: {row_rate:.4f}")

    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, config["data"]["train_file"]), index=False)
    val_df.to_csv(os.path.join(processed_dir, config["data"]["val_file"]), index=False)
    test_df.to_csv(os.path.join(processed_dir, config["data"]["test_file"]), index=False)
    print(f"\nSaved train/val/test CSVs to {processed_dir}/")
    print("(data_loader.py will auto-convert these to Parquet on first load)")


if __name__ == "__main__":
    main()