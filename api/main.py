"""FastAPI service for the two-stage QSR review classifier.

Run:  uvicorn api.main:app --host 127.0.0.1 --port 8000
Env:  MODELS_DIR (default ./models), COMPLAINT_THRESHOLD, CATEGORY_MIN_CONFIDENCE,
      API_KEY (optional; if set, requests must send header X-API-Key: <value>)

SECURITY: there is no auth unless API_KEY is set. Keep the default 127.0.0.1 bind unless you
set API_KEY and put the service behind TLS.
"""
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.config import MODELS_DIR
from src.inference import QSRPipeline, Thresholds

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(os.environ.get("MODELS_DIR", MODELS_DIR))
    state["pipeline"] = QSRPipeline.from_dir(models_dir, Thresholds())  # fails fast if models missing
    yield
    state.clear()


app = FastAPI(title="System One - QSR Review Classifier", version="0.2.0", lifespan=lifespan)


def check_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class Review(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    brand: Optional[str] = Field(default=None, max_length=100)  # passed through, not used by the model


class BatchReviews(BaseModel):
    reviews: list[Review] = Field(min_length=1, max_length=64)


@app.get("/health")
def health():
    p = state.get("pipeline")
    return {"status": "ok" if p else "loading", "thresholds": vars(p.th) if p else None}


@app.post("/api/v1/classify-review", dependencies=[Depends(check_key)])
def classify_review(review: Review):  # sync def -> runs in threadpool, doesn't block event loop
    out = state["pipeline"].classify([review.text])[0]
    return {"status": "success", "brand": review.brand, **out}


@app.post("/api/v1/classify-batch", dependencies=[Depends(check_key)])
def classify_batch(batch: BatchReviews):
    outs = state["pipeline"].classify([r.text for r in batch.reviews])
    return {"status": "success", "results": [{"brand": r.brand, **o} for r, o in zip(batch.reviews, outs)]}
