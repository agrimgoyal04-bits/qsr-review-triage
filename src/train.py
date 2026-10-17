"""Fine-tune DeBERTa-v3-small for either pipeline stage (CUDA, Apple MPS or CPU).

    python -m src.train --stage gatekeeper
    python -m src.train --stage router

Writes models/<stage>_standard/ (HF format, label maps stored in config.json)
and models/<stage>_standard/metrics.json (test-set metrics, per-class report, confusion matrix).

Device is auto-selected by HF Trainer: CUDA > Apple MPS > CPU. Training runs in fp32 on
purpose (DeBERTa-v3 + fp16/MPS is numerically fragile).
"""
import argparse
import gc
import json
import math
import os
import shutil
import sys
import time

# Must be set BEFORE torch is imported. Only affects Apple GPUs (MPS); harmless elsewhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported GPU ops fall back to CPU instead of crashing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoids noisy fork warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from .config import BASE_MODEL, DATA_DIR, MAX_LENGTH, MODELS_DIR, STAGES, id2label, label2id
from .slowdown import SlowdownDetector


def _mps_mem() -> str:
    try:
        return (f"allocated {torch.mps.current_allocated_memory() / 1e9:.1f} GB, "
                f"driver {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")
    except Exception:  # noqa: BLE001  (never let diagnostics break training)
        return "memory stats unavailable"


class SpeedWatch(TrainerCallback):
    """Guards against the 'fast at first, then slower and slower' problem seen on Apple GPUs (MPS).

    * every 25 steps on MPS: free cached GPU memory (gc + torch.mps.empty_cache)
    * every 100 steps on MPS: print a [mem] line (useful for diagnosing)
    * on any device: print a clear warning if steps become >4x slower than at the start
    """

    def __init__(self, on_mps: bool):
        self.on_mps = on_mps
        self.det = SlowdownDetector()

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if self.on_mps and step % 25 == 0:
            try:
                gc.collect()
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        if self.on_mps and step % 100 == 0:
            print(f"\n[mem] step {step}: {_mps_mem()}", flush=True)
        msg = self.det.update(step, time.time())
        if msg:
            print(msg + (f"\n[mem] step {step}: {_mps_mem()}" if self.on_mps else ""), flush=True)
        return control


class StopOnNaN(TrainerCallback):
    """Abort early with a helpful message if the loss becomes NaN (seen with some GPU backends)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and (isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss))):
            print(
                "\n*** Loss became NaN/inf - training is broken on this device. ***\n"
                "Try again with:  --cpu  (slower but safe)  and/or a smaller --lr (e.g. 1e-5).\n"
            )
            control.should_training_stop = True
            state.nan_detected = True
        return control


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        # No padding here: DataCollatorWithPadding pads per batch (much faster than max_length padding)
        self.enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def load_split(stage: str, split: str, limit: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{STAGES[stage]['data']}_{split}.parquet")
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=seed)
    return df.reset_index(drop=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


class WeightedTrainer(Trainer):
    """Optional class-weighted CE loss (useful for the imbalanced router)."""

    class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--pad-multiple", type=int, default=64,
                    help="pad each batch up to a multiple of this many tokens. Keeps the number of distinct tensor "
                         "shapes small (4 with the default), which avoids memory build-up/slowdown on Apple GPUs")
    ap.add_argument("--max-train", type=int, default=None, help="subsample train rows (dev/smoke runs)")
    ap.add_argument("--max-eval", type=int, default=None, help="subsample val/test rows")
    ap.add_argument("--class-weights", action="store_true", help="inverse-frequency loss weights")
    ap.add_argument("--out", default=None, help="override output dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="force CPU (slow but safe if the GPU/MPS misbehaves)")
    ap.add_argument("--keep-checkpoints", action="store_true", help="keep the big intermediate checkpoint folder")
    args = ap.parse_args()

    set_seed(args.seed)
    labels = STAGES[args.stage]["labels"]
    out_dir = MODELS_DIR / f"{args.stage}_standard" if args.out is None else __import__("pathlib").Path(args.out)
    ckpt_dir = MODELS_DIR / f"{args.stage}_checkpoints"

    train_df = load_split(args.stage, "train", args.max_train, args.seed)
    val_df = load_split(args.stage, "val", args.max_eval, args.seed)
    test_df = load_split(args.stage, "test", args.max_eval, args.seed)
    print(f"[{args.stage}] train={len(train_df)} val={len(val_df)} test={len(test_df)} classes={labels}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label(labels), label2id=label2id(labels),
        dtype=torch.float32,  # transformers>=5 defaults to dtype="auto" (checkpoint dtype, fp16 here) - force fp32
    )
    train_ds = TextDataset(train_df, tok, args.max_length)
    val_ds = TextDataset(val_df, tok, args.max_length)
    test_ds = TextDataset(test_df, tok, args.max_length)

    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=0.06,  # fraction of total steps
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        use_cpu=args.cpu,
    )
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=args.pad_multiple),
        compute_metrics=compute_metrics,
        callbacks=[StopOnNaN()],
    )
    dev = str(trainer.args.device)
    trainer.add_callback(SpeedWatch(on_mps=(dev == "mps")))
    print(f"\n>>> Training device: {dev.upper()}", flush=True)
    if dev == "cpu" and not args.cpu:
        print(
            ">>> WARNING: no GPU found, training on CPU. This will be MUCH slower.\n"
            ">>> PyTorch cannot see a GPU (CUDA on NVIDIA, MPS on Apple Silicon). Run: python scripts/doctor.py",
            flush=True,
        )
    if args.class_weights:
        counts = np.bincount(train_df["label"], minlength=len(labels)).astype(float)
        w = counts.sum() / (len(labels) * np.maximum(counts, 1))
        trainer.class_weights = torch.tensor(w, dtype=torch.float)
        print("class weights:", {l: round(float(x), 2) for l, x in zip(labels, w)})

    trainer.train()
    if getattr(trainer.state, "nan_detected", False):
        sys.exit("Training stopped because the loss became NaN. Nothing was saved. See message above.")

    # ---- final evaluation on held-out TEST split (businesses never seen in training) ----
    pred = trainer.predict(test_ds)
    y_true, y_pred = pred.label_ids, pred.predictions.argmax(-1)
    metrics = {
        "stage": args.stage,
        "base_model": args.model,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": labels,
        "per_class": classification_report(
            y_true, y_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
        "args": vars(args),
        "label_note": "Labels are weak (stars/keywords). See README 'Caveats'.",
    }

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nTEST accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print(f"Saved model + metrics.json to {out_dir}")
    if not args.keep_checkpoints and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)  # ~1.7 GB of optimizer state we no longer need
        print(f"Removed temporary checkpoints ({ckpt_dir.name})")


if __name__ == "__main__":
    main()
