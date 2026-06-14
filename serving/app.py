# serving/app.py
"""
FastAPI ML inference server.
Endpoints:
  POST /predict  — run inference
  GET  /metrics  — Prometheus metrics
  GET  /health   — liveness check
"""
from exporter.metrics import feature_removed
from exporter.metrics import feature_added
from exporter.metrics import datalake_unavailable
from exporter.metrics import distribution_drift_detected
import os
import time
import logging
import joblib
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Import shared metrics ──────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from exporter.metrics import model_accuracy, response_delay_seconds, retrain_count_total

# ── Model loading ──────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_PATH", "/app/model"))

def load_latest_model():
    latest_file = Path("/app/model/latest.txt")
    if latest_file.exists():
        # Read the relative path and resolve it from /app
        relative = latest_file.read_text().strip()
        model_path = Path("/app") / relative
    else:
        pkls = sorted(Path("/app/model").glob("model_v*.pkl"))
        if not pkls:
            raise FileNotFoundError("No model file found")
        model_path = pkls[-1]
    log.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


# Global model holder
app_state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["model"] = load_latest_model()
    # Load actual accuracy instead of defaulting to 0
    acc_file = Path("/app/model/latest_accuracy.txt")
    if acc_file.exists():
        model_accuracy.set(float(acc_file.read_text().strip()))
    else:
        model_accuracy.set(0.0)

    # Load actual retrain count instead of defaulting to 0
    retrain_file = Path("/app/model/latest_retrain_count.txt")
    if retrain_file.exists():
        try:
            count = float(retrain_file.read_text().strip())
            retrain_count_total.inc(count)
        except ValueError:
            pass
    yield


app = FastAPI(title="MLOps Inference API", lifespan=lifespan)


# ── Request / Response schemas ─────────────────────────────────────
class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: int | str
    confidence: float


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Accept a feature vector and return prediction + confidence.
    Measures latency and records it in the histogram.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        X = np.array(req.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        probas = model.predict_proba(X)[0]
        confidence = float(np.max(probas))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        latency = time.time() - start
        response_delay_seconds.observe(latency)

    return PredictResponse(prediction=int(prediction), confidence=confidence)


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics in text format."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/test/trigger-alerts")
def trigger_alerts():
    """Temporary endpoint to force alert conditions for testing."""
    model_accuracy.set(0.5)
    distribution_drift_detected.set(1)
    datalake_unavailable.inc()
    feature_added.inc()
    feature_removed.inc()
    retrain_count_total.inc()
    for _ in range(20):
        response_delay_seconds.observe(2.0)
    return {"status": "alert conditions set"}