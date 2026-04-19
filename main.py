"""
Fraud Detection API — main.py
Deploy on Render: uvicorn main:app --host 0.0.0.0 --port 10000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import os
import time

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time banking transaction fraud scoring using a neural network.",
    version="1.0.0"
)

# Allow requests from any origin (required for Replit frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ─────────────────────────────────────────────────────────────

model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = "fraud_model.h5"
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model file '{model_path}' not found. "
            "Upload fraud_model.h5 to the same folder as main.py."
        )
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Input shape: {model.input_shape}")


# ── Request / Response Schemas ────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    30 features expected by the model:
    - Time:    seconds elapsed from first transaction in dataset (scaled)
    - V1-V28: PCA-transformed anonymised features
    - Amount:  transaction amount in EUR (will be scaled server-side)

    For demo / testing you can send any 30 float values.
    """
    Time:   float = Field(..., example=-1.35)
    V1:     float = Field(..., example=-1.35)
    V2:     float = Field(..., example=-0.07)
    V3:     float = Field(..., example=2.54)
    V4:     float = Field(..., example=1.38)
    V5:     float = Field(..., example=-0.34)
    V6:     float = Field(..., example=0.46)
    V7:     float = Field(..., example=0.24)
    V8:     float = Field(..., example=0.10)
    V9:     float = Field(..., example=0.36)
    V10:    float = Field(..., example=0.09)
    V11:    float = Field(..., example=-0.55)
    V12:    float = Field(..., example=-0.62)
    V13:    float = Field(..., example=-0.99)
    V14:    float = Field(..., example=-0.31)
    V15:    float = Field(..., example=1.47)
    V16:    float = Field(..., example=-0.47)
    V17:    float = Field(..., example=0.21)
    V18:    float = Field(..., example=0.03)
    V19:    float = Field(..., example=0.40)
    V20:    float = Field(..., example=0.25)
    V21:    float = Field(..., example=-0.02)
    V22:    float = Field(..., example=0.28)
    V23:    float = Field(..., example=-0.11)
    V24:    float = Field(..., example=0.07)
    V25:    float = Field(..., example=0.13)
    V26:    float = Field(..., example=-0.19)
    V27:    float = Field(..., example=0.13)
    V28:    float = Field(..., example=-0.02)
    Amount: float = Field(..., example=149.62)


class PredictionResponse(BaseModel):
    fraud_probability: float   # 0.0 to 1.0
    verdict:           str     # "APPROVED" | "REVIEW" | "BLOCKED"
    risk_level:        str     # "LOW" | "MEDIUM" | "HIGH"
    confidence_pct:    float   # percentage confidence in the verdict
    latency_ms:        float   # inference time in milliseconds
    thresholds_used:   dict    # the threshold config applied


# ── Threshold Config ──────────────────────────────────────────────────────────

THRESHOLDS = {
    "approve_below": 0.40,   # score < 0.40  → APPROVED
    "review_below":  0.75,   # 0.40–0.75     → REVIEW
    # score >= 0.75           → BLOCKED
}


def score_to_verdict(score: float) -> tuple[str, str]:
    if score < THRESHOLDS["approve_below"]:
        return "APPROVED", "LOW"
    elif score < THRESHOLDS["review_below"]:
        return "REVIEW", "MEDIUM"
    else:
        return "BLOCKED", "HIGH"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Build feature vector in the exact order the model expects
    features = np.array([[
        tx.Time,
        tx.V1,  tx.V2,  tx.V3,  tx.V4,  tx.V5,  tx.V6,  tx.V7,
        tx.V8,  tx.V9,  tx.V10, tx.V11, tx.V12, tx.V13, tx.V14,
        tx.V15, tx.V16, tx.V17, tx.V18, tx.V19, tx.V20, tx.V21,
        tx.V22, tx.V23, tx.V24, tx.V25, tx.V26, tx.V27, tx.V28,
        tx.Amount
    ]], dtype=np.float32)

    # Run inference and measure latency
    t0 = time.perf_counter()
    prob = float(model.predict(features, verbose=0)[0][0])
    latency = (time.perf_counter() - t0) * 1000

    verdict, risk = score_to_verdict(prob)

    # Confidence = how far the score is from the nearest threshold boundary
    if verdict == "APPROVED":
        conf = (1.0 - prob / THRESHOLDS["approve_below"]) * 100
    elif verdict == "BLOCKED":
        conf = ((prob - THRESHOLDS["review_below"]) /
                (1.0 - THRESHOLDS["review_below"])) * 100
    else:
        mid = (THRESHOLDS["approve_below"] + THRESHOLDS["review_below"]) / 2
        conf = max(0, (1.0 - abs(prob - mid) / mid) * 100)

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        verdict=verdict,
        risk_level=risk,
        confidence_pct=round(max(0, min(100, conf)), 1),
        latency_ms=round(latency, 2),
        thresholds_used=THRESHOLDS
    )


@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionRequest]):
    """Score up to 100 transactions in one call."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100.")

    results = []
    for tx in transactions:
        results.append(predict(tx))
    return results


# ── Run locally ───────────────────────────────────────────────────────────────
# Run with:  uvicorn main:app --reload --port 8000
# Then open: http://localhost:8000/docs
