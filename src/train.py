"""
Training script for the Sepsis Detection project.
Trains and compares RandomForest and XGBoost classifiers with
class-imbalance handling, selects the better one by val AUPRC.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from data_loader import load_and_validate
from features import engineer_features


def get_feature_columns(train_feat, config) -> list[str]:
    exclude = {
        config["columns"]["id_col"],
        config["columns"]["time_col"],
        config["columns"]["label_col"],
    }
    return [c for c in train_feat.columns if c not in exclude]


def train_xgb(X_train, y_train, X_val, y_val, model_cfg):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=4,
        learning_rate=model_cfg["learning_rate"],
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=model_cfg["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_rf(X_train, y_train, model_cfg):
    model = RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=model_cfg["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_split(model, X, y, name):
    proba = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    print(f"{name} — AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    return auroc, auprc


def main():
    train_df, val_df, test_df, config = load_and_validate()

    print("\nEngineering features...")
    train_feat, fill_values = engineer_features(train_df, config)
    val_feat, _ = engineer_features(val_df, config, fill_values=fill_values)
    test_feat, _ = engineer_features(test_df, config, fill_values=fill_values)

    feature_cols = get_feature_columns(train_feat, config)
    label_col = config["columns"]["label_col"]
    print(f"Using {len(feature_cols)} features")

    X_train, y_train = train_feat[feature_cols], train_feat[label_col]
    X_val, y_val = val_feat[feature_cols], val_feat[label_col]
    X_test, y_test = test_feat[feature_cols], test_feat[label_col]

    candidates = {}

    print("\n--- Training XGBoost ---")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val, config["model"])
    _, xgb_val_auprc = evaluate_split(xgb_model, X_val, y_val, "XGBoost Val")
    candidates["xgboost"] = (xgb_model, xgb_val_auprc)

    print("\n--- Training RandomForest ---")
    rf_model = train_rf(X_train, y_train, config["model"])
    _, rf_val_auprc = evaluate_split(rf_model, X_val, y_val, "RandomForest Val")
    candidates["random_forest"] = (rf_model, rf_val_auprc)

    best_name = max(candidates, key=lambda k: candidates[k][1])
    best_model, best_val_auprc = candidates[best_name]
    print(f"\nBest model: {best_name} (val AUPRC={best_val_auprc:.4f})")

    print(f"\n--- Final evaluation: {best_name} ---")
    evaluate_split(best_model, X_train, y_train, "Train")
    evaluate_split(best_model, X_val, y_val, "Val")
    evaluate_split(best_model, X_test, y_test, "Test")

    os.makedirs(config["paths"]["model_dir"], exist_ok=True)
    model_path = os.path.join(config["paths"]["model_dir"], config["paths"]["model_name"])
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "model_type": best_name,
                "fill_values": fill_values,
                "feature_cols": feature_cols,
            },
            f,
        )
    print(f"\nSaved best model ({best_name}) + metadata to {model_path}")


if __name__ == "__main__":
    main()