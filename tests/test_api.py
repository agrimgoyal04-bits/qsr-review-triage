"""API tests. Skipped automatically if ONNX models are not present (models/*_onnx)."""
import os

import pytest
from fastapi.testclient import TestClient

from src.config import MODELS_DIR

MODELS = os.environ.get("MODELS_DIR", MODELS_DIR)
pytestmark = pytest.mark.skipif(
    not (os.path.exists(f"{MODELS}/gatekeeper_onnx/model.onnx") and os.path.exists(f"{MODELS}/router_onnx/model.onnx")),
    reason="ONNX models not exported yet",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_single(client):
    r = client.post("/api/v1/classify-review", json={"text": "Waited 30 minutes and the fries were cold.", "brand": "X"})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] in ("COMPLAINT", "OK") and 0 <= d["complaint_probability"] <= 1 and d["brand"] == "X"


def test_batch(client):
    r = client.post("/api/v1/classify-batch", json={"reviews": [{"text": "Great food"}, {"text": "Terrible rude staff"}]})
    assert r.status_code == 200 and len(r.json()["results"]) == 2


@pytest.mark.parametrize("payload", [{"text": ""}, {}, {"text": "x" * 20001}])
def test_validation(client, payload):
    assert client.post("/api/v1/classify-review", json=payload).status_code == 422


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/api/v1/classify-review", json={"text": "hi"}).status_code == 401
    assert client.post("/api/v1/classify-review", json={"text": "hi"}, headers={"X-API-Key": "bad"}).status_code == 401
    assert client.post("/api/v1/classify-review", json={"text": "hi"}, headers={"X-API-Key": "secret"}).status_code == 200
