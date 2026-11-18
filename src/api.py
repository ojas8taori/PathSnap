"""FastAPI entry point. Set PATHSNAP_DATA_ARCHIVE and PATHSNAP_ARTIFACT."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import PathSnapEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("pathsnap.api")
app = FastAPI(title="PathSnap", version="0.1.0")
origins = [item.strip() for item in os.getenv("PATHSNAP_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8502,http://127.0.0.1:8502").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
engine: PathSnapEngine | None = None


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")


@app.on_event("startup")
def load_once() -> None:
    global engine
    archive = os.getenv("PATHSNAP_DATA_ARCHIVE", "C:/OJ/sem5/ML/hackathon/Student folder.zip")
    artifact = os.getenv("PATHSNAP_ARTIFACT", "artifacts/final_model.joblib")
    partition = os.getenv("PATHSNAP_EVALUATOR_PARTITION", "evaluator_test")
    if Path(artifact).exists() and Path(archive).exists():
        engine = PathSnapEngine(archive, artifact, partition)
        logger.info("PathSnap ready; evaluator partition=%s trips=%d", partition, len(engine.available_trip_ids()))
    else:
        logger.error("PathSnap is not ready; missing archive or model artifact")


@app.get("/health")
def health() -> dict:
    return {"status": "ready" if engine is not None else "not_ready"}


@app.get("/trips")
def trips() -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model or evaluator archive is not configured")
    return {"trip_ids": engine.available_trip_ids()}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model or evaluator archive is not configured")
    if request.trip_id not in engine.available_trip_ids():
        raise HTTPException(status_code=404, detail="Trip ID is not in the evaluator partition")
    try:
        result = engine.predict(request.trip_id)
    except Exception as exc:
        logger.exception("Prediction failed for trip_id=%s", request.trip_id)
        raise HTTPException(status_code=500, detail="Prediction failed; check API host logs") from exc
    logger.info("prediction_complete trip_id=%s observations=%d metrics=%s", request.trip_id, len(result["predicted_arc_ids"]), result["metrics"])
    return result
