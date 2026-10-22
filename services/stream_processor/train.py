# -*- coding: utf-8 -*-
"""
Train XGBoost + LightGBM soft-voting ensemble
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from featurizer import FEATURE_NAMES


# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

CSV = os.path.join(BASE_DIR, "data", "creditcard.csv")

MODEL_OUT = os.path.join(BASE_DIR, "services", "stream_processor", "model.pkl")
META_OUT = MODEL_OUT.replace(".pkl", "_meta.json")


# ================= LOAD =================
def load(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    X = df[FEATURE_NAMES].values
    y = df["Class"].values

    return X, y


# ================= BEST THRESHOLD =================
def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

    best_idx = int(np.argmax(f1_scores[:-1]))

    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ================= MAIN =================
if __name__ == "__main__":
    print("\n Loading dataset...")
    X, y = load(CSV)

    print(" Dataset loaded")

    # ================= SPLIT =================
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    # ================= HANDLE IMBALANCE =================
    scale_pos = float((len(y_tr) - y_tr.sum()) / y_tr.sum())

    print(f" scale_pos_weight = {scale_pos:.2f}")
    print(f"Fraud rate = {y_tr.mean() * 100:.4f}%")

    # ================= MODELS =================
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        verbosity=0,
        random_state=42,
    )

    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        verbose=-1,
        random_state=42,
    )

    # ================= ENSEMBLE =================
    model = VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
        weights=[0.7, 0.3],
    )

    print("\n Training model...")
    model.fit(X_tr, y_tr)

    print(" Training completed")

    # ================= PREDICTIONS =================
    proba = model.predict_proba(X_te)[:, 1]

    pr_auc = average_precision_score(y_te, proba)

    best_thresh, best_f1 = best_f1_threshold(y_te, proba)

    y_pred = (proba >= best_thresh).astype(int)

    # ================= METRICS =================
    print("\n TEST METRICS")
    print("Accuracy :", accuracy_score(y_te, y_pred))
    print("Precision:", precision_score(y_te, y_pred))
    print("Recall   :", recall_score(y_te, y_pred))
    print("F1 Score :", f1_score(y_te, y_pred))
    print("PR-AUC   :", pr_auc)
    print("Threshold:", best_thresh)

    print("\n TRAIN METRICS")
    y_train_pred = model.predict(X_tr)

    print("Accuracy :", accuracy_score(y_tr, y_train_pred))
    print("Precision:", precision_score(y_tr, y_train_pred))
    print("Recall   :", recall_score(y_tr, y_train_pred))
    print("F1 Score :", f1_score(y_tr, y_train_pred))

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_te, y_pred)

    print("\n Confusion Matrix")
    print(cm)

    # ================= SAVE =================
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    joblib.dump(model, MODEL_OUT)
    print(f"\n Model saved → {MODEL_OUT}")

    meta = {
        "model": "VotingClassifier(XGB + LGBM)",
        "features": FEATURE_NAMES,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "pr_auc": round(pr_auc, 4),
        "f1": round(best_f1, 4),
        "threshold": round(best_thresh, 4),
        "fraud_rate_pct": round(float(y_tr.mean() * 100), 4),
    }

    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f" Meta saved → {META_OUT}")

    print("\n DONE")