import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"

LABELS = {
    -1: "Négatif",
    0: "Neutre",
    1: "Positif",
}

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage de l'API."""
    global model

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("Pipeline chargé avec succès.")
    else:
        print("Modèle introuvable :", MODEL_PATH)

    yield


app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API de classification des sentiments",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchRequest(BaseModel):
    comments: List[str] = Field(min_length=1, max_length=100)


@app.get("/")
def root():
    return {
        "message": "YouTube Sentiment Analysis API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé.",
        )

    comments = [comment.strip() for comment in request.comments]

    if any(not comment for comment in comments):
        raise HTTPException(
            status_code=400,
            detail="Un commentaire est vide.",
        )

    try:
        start = time.perf_counter()
        probabilities = model.predict_proba(comments)
        predictions = model.classes_[probabilities.argmax(axis=1)]
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        results = []
        for comment, prediction, proba in zip(comments, predictions, probabilities):
            results.append({
                "comment": comment,
                "label": int(prediction),
                "sentiment": LABELS[int(prediction)],
                "confidence": round(float(proba.max()), 4),
            })

        total = len(results)

        distribution = {
            "negative": sum(r["label"] == -1 for r in results),
            "neutral": sum(r["label"] == 0 for r in results),
            "positive": sum(r["label"] == 1 for r in results),
        }

        percentages = {
            key: round(value / total * 100, 2)
            for key, value in distribution.items()
        }

        return {
            "total_comments": total,
            "results": results,
            "distribution": distribution,
            "percentages": percentages,
            "processing_time_ms": elapsed_ms,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )