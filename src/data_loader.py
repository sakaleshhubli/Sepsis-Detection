"""
Data loading utilities for the Sepsis Detection project.
Loads train/val/test splits (converting to Parquet on first run for
faster, dtype-safe loading), and runs basic data quality checks.
"""

import os
import yaml
import pandas as pd


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _csv_to_parquet_if_needed(csv_path: str, parquet_path: str) -> pd.DataFrame:
    """
    Load from parquet if it exists and is newer than the CSV.
    Otherwise convert CSV -> parquet and return the DataFrame.
    """
    needs_conversion = (
        not os.path.exists(parquet_path)
        or os.path.getmtime(csv_path) > os.path.getmtime(parquet_path)
    )

    if needs_conversion:
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"Converted {csv_path} -> {parquet_path} ({df.shape[0]} rows)")
    else:
        df = pd.read_parquet(parquet_path, engine="pyarrow")

    return df


def load_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, val, and test data from data/processed as defined in config.
    CSVs are converted to Parquet on first run and re-loaded from Parquet
    thereafter (auto re-converts if the CSV is newer than the cached file).

    Returns:
        (train_df, val_df, test_df)
    """
    processed_dir = config["data"]["processed_dir"]

    dfs = []
    for split_file in [
        config["data"]["train_file"],
        config["data"]["val_file"],
        config["data"]["test_file"],
    ]:
        csv_path = os.path.join(processed_dir, split_file)
        parquet_path = os.path.join(
            processed_dir, split_file.replace(".csv", ".parquet")
        )
        dfs.append(_csv_to_parquet_if_needed(csv_path, parquet_path))

    train_df, val_df, test_df = dfs
    return train_df, val_df, test_df


def data_quality_report(df: pd.DataFrame, name: str, config: dict) -> None:
    """Print a quick data quality summary for a given split."""
    label_col = config["columns"]["label_col"]
    id_col = config["columns"]["id_col"]

    print(f"\n{'='*50}")
    print(f"Data Quality Report: {name}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Unique patients: {df[id_col].nunique()}")
    print(f"Sepsis rate: {df[label_col].mean():.4f}")
    print(f"\nMissingness by column (%):")
    missing_pct = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(missing_pct)


def load_and_validate(config_path: str = "configs/config.yaml"):
    """Convenience function: load config + splits, print quality reports."""
    config = load_config(config_path)
    train_df, val_df, test_df = load_splits(config)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        data_quality_report(df, name, config)

    return train_df, val_df, test_df, config


if __name__ == "__main__":
    load_and_validate()