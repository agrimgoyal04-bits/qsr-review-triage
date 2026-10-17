"""Single source of truth for paths, model name and label sets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"

BASE_MODEL = "microsoft/deberta-v3-small"
MAX_LENGTH = 256  # tokens; reviews are mostly shorter, long ones get truncated

# Stage 1 - Gatekeeper: does this review need attention?
GATE_LABELS = ["not_complaint", "complaint"]  # index == class id

# Stage 2 - Router: what is the complaint about? (weak keyword labels, see qsr_labels.py)
ROUTER_LABELS = [
    "order_accuracy",
    "wait_time_service_speed",
    "food_quality",
    "cleanliness_hygiene",
    "staff_attitude",
    "price_value",
    "other",
]

STAGES = {
    "gatekeeper": {"labels": GATE_LABELS, "data": "gatekeeper"},
    "router": {"labels": ROUTER_LABELS, "data": "router"},
}


def id2label(labels):
    return {i: l for i, l in enumerate(labels)}


def label2id(labels):
    return {l: i for i, l in enumerate(labels)}
