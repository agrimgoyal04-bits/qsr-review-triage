import numpy as np
import pandas as pd
import pytest

from src.build_dataset import split_of
from src.config import GATE_LABELS, ROUTER_LABELS, id2label
from src.inference import QSRPipeline, Thresholds, softmax
from src.qsr_labels import normalize_brand, sentiment_from_stars, weak_category


# ---------- weak labels ----------
@pytest.mark.parametrize("stars,expected", [(1, 1), (2, 1), (3, None), (4, 0), (5, 0)])
def test_sentiment_from_stars(stars, expected):
    assert sentiment_from_stars(stars) == expected


@pytest.mark.parametrize(
    "text,cat",
    [
        ("They gave me the wrong order and forgot my sauce", "order_accuracy"),
        ("We waited 40 minutes in the drive thru, so slow", "wait_time_service_speed"),
        ("The fries were cold and soggy and tasteless", "food_quality"),
        ("The bathroom was filthy and the tables were dirty", "cleanliness_hygiene"),
        ("The cashier was so rude and unprofessional", "staff_attitude"),
        ("Totally overpriced, not worth the money", "price_value"),
        ("I went on a Tuesday.", "other"),
    ],
)
def test_weak_category(text, cat):
    assert weak_category(text) == cat
    assert cat in ROUTER_LABELS


def test_normalize_brand():
    assert normalize_brand("Dairy Queen Grill & Chill") == "Dairy Queen"
    assert normalize_brand("McDonald's") == "McDonald's"
    assert normalize_brand("Some Local Place") == "Some Local Place"


# ---------- split ----------
def test_split_is_deterministic_and_grouped():
    ids = [f"biz{i}" for i in range(2000)]
    s1 = [split_of(i) for i in ids]
    assert s1 == [split_of(i) for i in ids]
    frac = {k: s1.count(k) / len(s1) for k in ("train", "val", "test")}
    assert 0.75 < frac["train"] < 0.85 and 0.06 < frac["val"] < 0.14 and 0.06 < frac["test"] < 0.14


def test_built_datasets_have_no_business_leakage():
    from src.config import DATA_DIR

    files = {s: DATA_DIR / f"gatekeeper_{s}.parquet" for s in ("train", "val", "test")}
    if not all(f.exists() for f in files.values()):
        pytest.skip("dataset not built")
    biz = {s: set(pd.read_parquet(f, columns=["business_id"])["business_id"]) for s, f in files.items()}
    assert not (biz["train"] & biz["val"]) and not (biz["train"] & biz["test"]) and not (biz["val"] & biz["test"])


# ---------- pipeline logic with stub models ----------
class Stub:
    def __init__(self, id2l, fn):
        self.id2label, self.fn = id2l, fn

    def predict_proba(self, texts):
        return np.array([self.fn(t) for t in texts])


def make_pipeline(**th):
    gate = Stub(id2label(GATE_LABELS), lambda t: [0.1, 0.9] if "bad" in t else [0.9, 0.1])
    def router_fn(t):
        p = np.full(len(ROUTER_LABELS), 0.02)
        p[ROUTER_LABELS.index("food_quality" if "cold" in t else "price_value")] = 1 - 0.02 * (len(ROUTER_LABELS) - 1)
        return p
    return QSRPipeline(gate, Stub(id2label(ROUTER_LABELS), router_fn), Thresholds(**th))


def test_pipeline_cascade():
    out = make_pipeline().classify(["bad and cold", "great burger"])
    assert out[0]["decision"] == "COMPLAINT" and out[0]["category"] == "food_quality"
    assert abs(sum(out[0]["category_probabilities"].values()) - 1) < 0.01
    assert out[1]["decision"] == "OK" and out[1]["category"] is None


def test_pipeline_uncertain_below_confidence():
    out = make_pipeline(category_min_confidence=0.99).classify(["bad and cold"])
    assert out[0]["category"] == "uncertain"


def test_pipeline_threshold_configurable():
    assert make_pipeline(complaint=0.95).classify(["bad"])[0]["decision"] == "OK"


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(p.sum(-1), 1) and not np.isnan(p).any()
