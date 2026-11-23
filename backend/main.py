from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from model_pipeline import RAW_FEATURES, NotebookPklPreprocessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Cardiovascular_Risk_Prediction_Classification.pkl"
MODEL_VERSION = "vois-cardio-xgb-notebook-pkl-1.0.0"
DISCLAIMER = (
    "Educational demonstration only. This output is not a medical diagnosis, "
    "not a calibrated clinical probability, and must not guide treatment."
)

app = FastAPI(
    title="VOIS Cardiovascular Risk Prediction API",
    version=MODEL_VERSION,
    description="Academic XGBoost classification demonstration for TenYearCHD.",
)

origins = [x.strip() for x in os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,"
    "https://vois-nov-ai-ml.vercel.app",
).split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://vois-nov-ai-ml(-[a-z0-9-]+)?\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

model = None
pkl_preprocessor = None
if MODEL_PATH.exists():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    pkl_preprocessor = NotebookPklPreprocessor(BASE_DIR.parent / "data_cardiovascular_risk.csv")

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=18, le=110)
    education: float = Field(ge=1, le=4)
    sex: Literal["F", "M"]
    cigsPerDay: float = Field(ge=0, le=100)
    BPMeds: float = Field(ge=0, le=1)
    prevalentStroke: int = Field(ge=0, le=1)
    prevalentHyp: int = Field(ge=0, le=1)
    diabetes: int = Field(ge=0, le=1)
    totChol: float = Field(gt=0, le=1000)
    sysBP: float = Field(gt=0, le=400)
    diaBP: float = Field(gt=0, le=250)
    BMI: float = Field(gt=0, le=100)
    heartRate: float = Field(gt=0, le=250)
    glucose: float = Field(gt=0, le=1000)

@app.get("/")
def root():
    return {
        "service": "VOIS Cardiovascular Risk Prediction API",
        "status": "online",
        "model": "XGBClassifier",
        "model_version": MODEL_VERSION,
        "docs": "/docs",
    }

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact missing: Cardiovascular_Risk_Prediction_Classification.pkl."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "XGBClassifier",
        "model_version": MODEL_VERSION,
        "features": RAW_FEATURES,
    }

@app.post("/predict")
def predict(payload: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not found.")
    try:
        row = pd.DataFrame([payload.model_dump()])
        if pkl_preprocessor is None:
            raise HTTPException(status_code=503, detail="Model preprocessor not initialized.")
        model_input = pkl_preprocessor.transform(row)
        prediction = int(model.predict(model_input)[0])
        probabilities = model.predict_proba(model_input)[0]
        classes = list(model.classes_)
        p0 = float(probabilities[classes.index(0)])
        p1 = float(probabilities[classes.index(1)])
        return {
            "prediction": prediction,
            "prediction_label": "Positive class (TenYearCHD = 1)" if prediction == 1
                                else "Negative class (TenYearCHD = 0)",
            "probability_class_0": round(p0, 6),
            "probability_class_1": round(p1, 6),
            "threshold": 0.5,
            "model": "XGBClassifier",
            "model_version": MODEL_VERSION,
            "disclaimer": DISCLAIMER,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
