# serving/app.py
"""
FastAPI ML inference server.
Endpoints:
  POST /predict  — run inference
  GET  /metrics  — Prometheus metrics
  GET  /health   — liveness check
"""
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
from exporter.metrics import model_accuracy, response_delay_seconds

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
    """Load model at startup."""
    app_state["model"] = load_latest_model()
    log.info("Model loaded successfully")
    # Set initial accuracy metric (will be updated after retraining)
    model_accuracy.set(0.0)
    yield
    log.info("Shutting down")


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
