"""
FastAPI REST API — Live Anomaly Detection Endpoint
====================================================
Serves trained models via REST for real-time pipeline metric scoring.

Endpoints:
  POST /predict   — score a single pipeline run
  GET  /health    — health check
  GET  /results   — return latest model comparison results

Run locally:
    uvicorn azure.api:app --reload --port 8000

Deploy to Azure Container Apps (free tier: 180k vCPU-s / month free):
    az containerapp up --name anomaly-api --resource-group <rg> --source .
"""

import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title       = "CI/CD Pipeline Anomaly Detector",
    description = "AI-based anomaly detection for CI/CD pipeline metrics",
    version     = "1.0.0",
)

# ── Request / Response Models ────────────────────────────────────────────────
class PipelineRun(BaseModel):
    build_duration:       float = Field(..., ge=0, description="Build duration in seconds")
    test_execution_time:  float = Field(..., ge=0, description="Test execution time in seconds")
    deployment_frequency: float = Field(..., ge=0, description="Deployments per day")
    resource_utilization: float = Field(..., ge=0, le=100, description="Resource usage %")
    error_count:          int   = Field(default=0, ge=0)


class PredictionResponse(BaseModel):
    is_anomaly:      bool
    anomaly_score:   float
    anomaly_type:    str
    confidence:      str


# ── Lazy-load models ─────────────────────────────────────────────────────────
_if_detector    = None
_lstm_detector  = None
_ensemble       = None
_prep           = None


def _load_models():
    global _if_detector, _lstm_detector, _ensemble, _prep
    if _if_detector is not None:
        return

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.preprocessor import PipelinePreprocessor
    from src.models import (
        IsolationForestDetector, LSTMAutoencoderDetector, EnsembleDetector
    )

    _prep = PipelinePreprocessor()
    _if_detector   = IsolationForestDetector()
    _lstm_detector = LSTMAutoencoderDetector()
    _ensemble      = EnsembleDetector(_if_detector, _lstm_detector)

    try:
        _if_detector.load()
    except Exception:
        raise HTTPException(500, "Isolation Forest model not found. Run main.py first.")
    try:
        _lstm_detector.load()
    except Exception:
        raise HTTPException(500, "LSTM model not found. Run main.py first.")


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "CI/CD Anomaly Detector v1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(run: PipelineRun):
    _load_models()
    import pandas as pd

    row = pd.DataFrame([{
        "build_duration":       run.build_duration,
        "test_execution_time":  run.test_execution_time,
        "deployment_frequency": run.deployment_frequency,
        "resource_utilization": run.resource_utilization,
        "error_count":          run.error_count,
    }])

    X_scaled = _prep.transform(row)
    if_score = float(_if_detector.score_samples(X_scaled)[0])
    is_anom  = bool(_if_detector.predict(X_scaled)[0])

    # Classify type from heuristics
    if run.build_duration > 580:
        atype = "spike"
    elif run.deployment_frequency < 1.5:
        atype = "frequency_drop"
    elif if_score > 0.75:
        atype = "drift"
    else:
        atype = "normal"

    confidence = "high" if if_score > 0.8 else "medium" if if_score > 0.5 else "low"

    return PredictionResponse(
        is_anomaly    = is_anom,
        anomaly_score = round(if_score, 4),
        anomaly_type  = atype,
        confidence    = confidence,
    )


@app.get("/results")
def get_results():
    path = "results/model_results.json"
    if not os.path.exists(path):
        raise HTTPException(404, "No results found. Run main.py first.")
    with open(path) as f:
        return json.load(f)
