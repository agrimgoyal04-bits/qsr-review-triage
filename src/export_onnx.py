"""Export a trained stage to ONNX for fast CPU inference, and verify it matches PyTorch.

    python -m src.export_onnx --stage gatekeeper
    python -m src.export_onnx --stage router
    python -m src.export_onnx --src models/gatekeeper_standard --dst models/gatekeeper_onnx

Output dir contains model.onnx, tokenizer files, config.json (with id2label) and metrics.json.
The API loads these with onnxruntime (no torch needed at serving time).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MODELS_DIR


def export(src: Path, dst: Path, opset: int = 17, tol: float = 1e-3):
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32).eval()
    model.config.return_dict = True

    sample = tok(["The fries were cold and the staff was rude.", "ok"], return_tensors="pt", padding=True)
    dst.mkdir(parents=True, exist_ok=True)
    onnx_path = dst / "model.onnx"

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    with torch.no_grad():
        torch.onnx.export(
            Wrapper(model),
            (sample["input_ids"], sample["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,  # legacy tracer: most reliable for DeBERTa-v2 attention
        )

    tok.save_pretrained(dst)
    model.config.save_pretrained(dst)
    for extra in ("metrics.json",):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)

    # ---- verify ONNX == PyTorch on different batch/seq shapes than used for tracing ----
    # Reference model is re-loaded from disk: the traced instance can be left in a modified
    # state by torch.onnx.export (observed with DeBERTa), which would give a false mismatch.
    import onnxruntime as ort

    ref_model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32).eval()
    so = ort.SessionOptions()
    so.log_severity_level = 3  # silence harmless constant-folding warnings
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    texts = ["Waited 25 minutes in the drive thru and they forgot my drink.", "Great burger!", "x " * 100]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        ref = ref_model(**enc).logits.numpy()
    out = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    diff = float(np.abs(ref - out).max())
    print(f"max |torch - onnx| logits diff = {diff:.2e}")
    if diff > tol:
        raise SystemExit(f"ONNX output deviates from PyTorch (>{tol}); do not deploy this export.")
    print(f"OK: {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gatekeeper", "router", "both"], default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    a = ap.parse_args()
    if a.src and a.dst:
        export(Path(a.src), Path(a.dst))
    else:
        for st in (["gatekeeper", "router"] if a.stage in (None, "both") else [a.stage]):
            export(MODELS_DIR / f"{st}_standard", MODELS_DIR / f"{st}_onnx")
