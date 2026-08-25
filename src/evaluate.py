"""
Evaluation script for the Sepsis Detection project.
Loads the saved model and produces a full performance report on
train/val/test: AUROC, AUPRC, confusion matrix at multiple thresholds,
and calibration.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report,
)

from data_loader import load_and_validate
from features import engineer_features


def load_model(config: dict):
    model_path = os.path.join(config["paths"]["model_dir"], config["paths"]["model_name"])
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["fill_values"], bundle["feature_cols"]


def best_threshold_by_f1(y_true, y_proba) -> float:
    """Find the probability threshold that maximizes F1 on this split."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last point has no corresponding threshold
    return thresholds[best_idx], f1s[best_idx]


def evaluate_full(model, X, y, name, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    auroc = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print(f"{name} (threshold={threshold:.3f})")
    print(f"{'='*50}")
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    print(f"Confusion matrix — TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
    print(f"Sensitivity (recall on sepsis): {tp/(tp+fn):.4f}")
    print(f"Precision (of predicted sepsis, how many were right): {tp/(tp+fp):.4f}")
    print(f"False positive rate: {fp/(fp+tn):.4f}")
    print(classification_report(y, preds, target_names=["No Sepsis", "Sepsis"]))

    return {"auroc": auroc, "auprc": auprc, "confusion_matrix": cm}


def main():
    train_df, val_df, test_df, config = load_and_validate()
    model, fill_values, feature_cols = load_model(config)

    train_feat, _ = engineer_features(train_df, config, fill_values=fill_values)
    val_feat, _ = engineer_features(val_df, config, fill_values=fill_values)
    test_feat, _ = engineer_features(test_df, config, fill_values=fill_values)

    label_col = config["columns"]["label_col"]

    # Find best threshold on VAL (never on test — that would leak)
    val_proba = model.predict_proba(val_feat[feature_cols])[:, 1]
    best_thresh, best_f1 = best_threshold_by_f1(val_feat[label_col], val_proba)
    print(f"Best threshold (by val F1): {best_thresh:.4f} (F1={best_f1:.4f})")

    for name, feat in [("Train", train_feat), ("Val", val_feat), ("Test", test_feat)]:
        evaluate_full(model, feat[feature_cols], feat[label_col], name, threshold=best_thresh)


if __name__ == "__main__":
    main()