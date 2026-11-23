from __future__ import annotations

import json
from pathlib import Path
import sys
import platform
import sklearn
import imblearn
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from model_pipeline import RAW_FEATURES, NotebookFeatureEngineer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_cardiovascular_risk.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH)
    required = RAW_FEATURES + ["TenYearCHD"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    X = data[RAW_FEATURES].copy()
    y = data["TenYearCHD"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("TenYearCHD must contain both binary classes 0 and 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=10, stratify=y
    )

    pipeline = Pipeline(steps=[
        ("features", NotebookFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=10)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=10,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset_rows": int(len(data)),
        "dataset_columns": int(data.shape[1]),
        "train_rows_before_smote": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_class_prevalence": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, zero_division=0, output_dict=True
        ),
        "evaluation": "Stratified 80/20 holdout; imputation, transformations, scaling and SMOTE fitted using training data only.",
        "model": "RandomForestClassifier",
        "random_state": 10,
    }
    (MODEL_DIR / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Refit the same leakage-safe pipeline on all supplied rows for deployment.
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_DIR / "random_forest_pipeline.joblib", compress=3)
    metadata = {
        "model": "RandomForestClassifier",
        "model_version": "vois-cardio-rf-1.0.0",
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "imbalanced_learn_version": imblearn.__version__,
        "dataset_file": "data_cardiovascular_risk.csv",
        "feature_engineering": "NotebookFeatureEngineer",
        "training_rows": int(len(X)),
        "random_state": 10,
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("classification_report",)}, indent=2))
    print("Saved:", MODEL_DIR / "random_forest_pipeline.joblib")

if __name__ == "__main__":
    main()
