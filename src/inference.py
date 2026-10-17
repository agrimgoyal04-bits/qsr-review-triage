"""ONNX inference for the two-stage QSR review pipeline (no torch needed at serving time).

Stage 1 (gatekeeper): P(complaint). If >= complaint_threshold -> run Stage 2.
Stage 2 (router): complaint category. If top prob < category_min_confidence -> "uncertain".
Label names are read from each model's config.json (id2label), never hardcoded.
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MAX_LENGTH, MODELS_DIR


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class OnnxTextClassifier:
    def __init__(self, model_dir: Path, max_length: int = MAX_LENGTH, threads: Optional[int] = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        onnx_file = model_dir / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"{onnx_file} not found. Train + export first (see README).")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.max_length = max_length
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_file), so, providers=["CPUExecutionProvider"])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=self.max_length)
        logits = self.session.run(
            None,
            {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)},
        )[0]
        return softmax(logits)


@dataclass
class Thresholds:
    complaint: float = float(os.environ.get("COMPLAINT_THRESHOLD", 0.5))
    category_min_confidence: float = float(os.environ.get("CATEGORY_MIN_CONFIDENCE", 0.5))


class QSRPipeline:
    """gatekeeper/router can be any object with .predict_proba(list[str]) and .id2label (eases testing)."""

    def __init__(self, gatekeeper, router, thresholds: Optional[Thresholds] = None):
        self.gate, self.router = gatekeeper, router
        self.th = thresholds or Thresholds()
        self.complaint_id = next(i for i, l in self.gate.id2label.items() if l == "complaint")

    @classmethod
    def from_dir(cls, models_dir: Path = MODELS_DIR, thresholds: Optional[Thresholds] = None):
        return cls(
            OnnxTextClassifier(Path(models_dir) / "gatekeeper_onnx"),
            OnnxTextClassifier(Path(models_dir) / "router_onnx"),
            thresholds,
        )

    def classify(self, texts: list[str]) -> list[dict]:
        t0 = time.perf_counter()
        gate_p = self.gate.predict_proba(texts)[:, self.complaint_id]
        is_c = gate_p >= self.th.complaint
        results = [
            {
                "decision": "COMPLAINT" if c else "OK",
                "complaint_probability": round(float(p), 4),
                "category": None,
                "category_confidence": None,
                "category_probabilities": None,
            }
            for p, c in zip(gate_p, is_c)
        ]
        idx = [i for i, c in enumerate(is_c) if c]
        if idx:  # router only runs on flagged reviews (cascade saves compute)
            rp = self.router.predict_proba([texts[i] for i in idx])
            for row, i in zip(rp, idx):
                k = int(row.argmax())
                conf = float(row[k])
                results[i]["category"] = self.router.id2label[k] if conf >= self.th.category_min_confidence else "uncertain"
                results[i]["category_confidence"] = round(conf, 4)
                results[i]["category_probabilities"] = {
                    self.router.id2label[j]: round(float(row[j]), 4) for j in range(len(row))
                }
        per_item_ms = round((time.perf_counter() - t0) * 1000 / max(len(texts), 1), 2)
        for r in results:
            r["latency_ms"] = per_item_ms
        return results
