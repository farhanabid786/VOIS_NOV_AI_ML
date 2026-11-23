from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

RAW_FEATURES = [
    "age", "education", "sex", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]
MODEL_FEATURES = [
    "age", "education", "sex", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "BMI", "heartRate", "glucose", "pulsePressure",
]
LOG_COLUMNS = [
    "age", "cigsPerDay", "totChol", "sysBP", "diaBP",
    "BMI", "heartRate", "glucose",
]
MODE_COLUMNS = ["education", "BPMeds"]
MEDIAN_COLUMNS = [
    "age", "cigsPerDay", "totChol", "sysBP", "diaBP",
    "BMI", "heartRate", "glucose",
]

NOTEBOOK_MODEL_FEATURES = [
    "age", "education", "sex", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "BMI", "heartRate", "glucose", "pulsePressure",
]

class NotebookPklPreprocessor:
    """Recreates the preprocessing used when the uploaded .pkl was trained."""

    def __init__(self, dataset_path: Path):
        frame = pd.read_csv(dataset_path)
        frame = frame[RAW_FEATURES].copy()

        self.medians_ = {
            col: float(pd.to_numeric(frame[col], errors="coerce").median())
            for col in ("glucose", "totChol", "cigsPerDay", "BMI", "heartRate")
        }
        self.modes_ = {}
        for col in ("education", "BPMeds"):
            mode = pd.to_numeric(frame[col], errors="coerce").mode(dropna=True)
            self.modes_[col] = float(mode.iloc[0]) if not mode.empty else 0.0

        fitted = self._transform_unscaled(frame)
        self.scaler = StandardScaler()
        self.scaler.fit(fitted)

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=RAW_FEATURES)
        frame = self._transform_unscaled(frame)
        return self.scaler.transform(frame)

    def _transform_unscaled(self, frame):
        frame = frame.copy()
        for col, value in self.medians_.items():
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(value)
        for col, value in self.modes_.items():
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(value)

        raw_sysbp = pd.to_numeric(frame["sysBP"], errors="coerce").astype(float)
        raw_diabp = pd.to_numeric(frame["diaBP"], errors="coerce").astype(float)
        pulse_delta = raw_sysbp - raw_diabp
        if (pulse_delta < 0).any():
            raise ValueError("Systolic blood pressure must be >= diastolic blood pressure")
        frame["pulsePressure"] = np.log1p(pulse_delta)

        sex_values = frame["sex"].astype(str).str.strip().str.upper()
        sex_map = {"F": 0.0, "M": 1.0, "FEMALE": 0.0, "MALE": 1.0,
                   "0": 0.0, "1": 1.0}
        mapped = sex_values.map(sex_map)
        if mapped.isna().any():
            raise ValueError("sex must be F or M")
        frame["sex"] = mapped.astype(float)

        for col in ("age", "cigsPerDay", "totChol", "sysBP", "diaBP",
                    "BMI", "heartRate", "glucose"):
            values = pd.to_numeric(frame[col], errors="coerce").astype(float)
            if (values < 0).any():
                raise ValueError(f"{col} cannot contain negative values")
            frame[col] = np.log1p(values)
        return frame[NOTEBOOK_MODEL_FEATURES].astype(float)

class NotebookFeatureEngineer(BaseEstimator, TransformerMixin):
    """Reproduces notebook feature definitions while fitting imputers on train data only."""

    def fit(self, X, y=None):
        frame = self._frame(X)
        self.medians_ = {
            col: float(pd.to_numeric(frame[col], errors="coerce").median())
            for col in MEDIAN_COLUMNS
        }
        self.modes_ = {}
        for col in MODE_COLUMNS:
            mode = pd.to_numeric(frame[col], errors="coerce").mode(dropna=True)
            self.modes_[col] = float(mode.iloc[0]) if not mode.empty else 0.0
        return self

    @staticmethod
    def _frame(X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X, columns=RAW_FEATURES)
        missing = [c for c in RAW_FEATURES if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        return frame[RAW_FEATURES].copy()

    def transform(self, X):
        frame = self._frame(X)
        for col, value in self.medians_.items():
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(value)
        for col, value in self.modes_.items():
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(value)

        sex_map = {"F": 0.0, "M": 1.0, "FEMALE": 0.0, "MALE": 1.0,
                   "0": 0.0, "1": 1.0}
        sex_values = frame["sex"].astype(str).str.strip().str.upper()
        mapped = sex_values.map(sex_map)
        if mapped.isna().any():
            raise ValueError("sex must be F or M")
        frame["sex"] = mapped.astype(float)

        for col in LOG_COLUMNS:
            values = pd.to_numeric(frame[col], errors="coerce").astype(float)
            if (values < 0).any():
                raise ValueError(f"{col} cannot contain negative values")
            frame[col] = np.log1p(values)

        # Notebook: pulsePressure = log1p(log1p(sysBP) - log1p(diaBP)).
        pulse_delta = frame["sysBP"] - frame["diaBP"]
        if (pulse_delta < 0).any():
            raise ValueError("Systolic blood pressure must be >= diastolic blood pressure")
        frame["pulsePressure"] = np.log1p(pulse_delta)

        output = frame[MODEL_FEATURES].astype(float)
        if not np.isfinite(output.to_numpy()).all():
            raise ValueError("Input contains non-finite values")
        return output

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MODEL_FEATURES, dtype=object)
