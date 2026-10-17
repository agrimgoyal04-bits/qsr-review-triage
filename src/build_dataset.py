"""Build QSR (quick-service restaurant) review datasets from the Yelp Open Dataset.

Inputs  (auto-downloaded from Hugging Face into data/raw/ if missing, ~3 GB for reviews):
    yashraizad/yelp-open-dataset-business  -> business.parquet
    yashraizad/yelp-open-dataset-reviews   -> review.parquet
Outputs (data/):
    gatekeeper_{train,val,test}.parquet   text, label(0/1), review_id, business_id, brand, stars
    router_{train,val,test}.parquet       text, label(0..6), category, ...   (complaints only)
    gold_to_label.csv                     rows to hand-label (for honest router evaluation)
    dataset_stats.json

Splits are grouped by business_id so the same restaurant never appears in two splits.
Labels are WEAK (stars / keywords) - see src/qsr_labels.py.

Usage:
    python -m src.build_dataset                      # full Fast Food set
    python -m src.build_dataset --max-reviews 20000  # quick dev subset
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import DATA_DIR, RAW_DIR, ROUTER_LABELS, label2id
from .qsr_labels import normalize_brand, sentiment_from_stars, weak_category

BUSINESS_REPO = "yashraizad/yelp-open-dataset-business"
REVIEW_REPO = "yashraizad/yelp-open-dataset-reviews"
MIN_CHARS = 20


def ensure_raw(repo: str, filename: str) -> Path:
    path = RAW_DIR / filename
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from {repo} ...")
    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(RAW_DIR)))


def split_of(business_id: str, seed: str = "qsr-v1") -> str:
    """Deterministic group split by business: 80% train / 10% val / 10% test."""
    h = int(hashlib.md5(f"{seed}:{business_id}".encode()).hexdigest(), 16) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def load_qsr_businesses(category_match: str, min_locations: int) -> pd.DataFrame:
    b = pd.read_parquet(ensure_raw(BUSINESS_REPO, "business.parquet"))
    b = b[b["categories"].fillna("").str.contains(category_match, case=False, regex=False)].copy()
    b["brand"] = b["name"].map(normalize_brand)
    # keep chains only: independents tagged "Fast Food" on Yelp are not QSR brands
    counts = b.groupby("brand")["business_id"].transform("count")
    b = b[counts >= min_locations]
    return b[["business_id", "brand", "city", "state"]]


def load_reviews(business_ids: set, max_reviews: int | None) -> pd.DataFrame:
    pf = pq.ParquetFile(ensure_raw(REVIEW_REPO, "review.parquet"))
    parts, n = [], 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["review_id", "business_id", "date", "stars", "text"]).to_pandas()
        t = t[t["business_id"].isin(business_ids)]
        if len(t):
            parts.append(t)
            n += len(t)
        print(f"  row group {rg + 1}/{pf.num_row_groups}: kept so far {n}")
        if max_reviews and n >= max_reviews:
            break
    df = pd.concat(parts, ignore_index=True)
    return df.iloc[:max_reviews] if max_reviews else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category-match", default="Fast Food", help="substring matched in Yelp business categories")
    ap.add_argument("--min-locations", type=int, default=5, help="keep brands with at least this many locations")
    ap.add_argument("--max-reviews", type=int, default=None, help="cap rows (dev only)")
    ap.add_argument("--gold-size", type=int, default=600)
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    biz = load_qsr_businesses(args.category_match, args.min_locations)
    print(f"{len(biz)} QSR businesses, {biz.brand.nunique()} distinct brands")

    rev = load_reviews(set(biz.business_id), args.max_reviews)
    rev = rev.merge(biz, on="business_id", how="left")
    rev["text"] = rev["text"].map(clean_text)
    rev = rev[rev["text"].str.len() >= MIN_CHARS].drop_duplicates("text")
    rev["gate_label"] = rev["stars"].map(sentiment_from_stars)
    rev = rev.dropna(subset=["gate_label"]).copy()  # drops 3-star
    rev["gate_label"] = rev["gate_label"].astype(int)
    rev["split"] = rev["business_id"].map(split_of)

    keep = ["review_id", "business_id", "brand", "state", "stars", "text", "split"]

    # ---- gatekeeper ----
    gate = rev[keep + ["gate_label"]].rename(columns={"gate_label": "label"})
    # ---- router (complaints only) ----
    rt = rev[rev["gate_label"] == 1][keep].copy()
    rt["category"] = rt["text"].map(weak_category)
    rt["label"] = rt["category"].map(label2id(ROUTER_LABELS))

    stats = {"businesses": int(len(biz)), "brands": int(biz.brand.nunique()), "category_match": args.category_match}
    for name, df in (("gatekeeper", gate), ("router", rt)):
        for sp in ("train", "val", "test"):
            part = df[df["split"] == sp].drop(columns=["split"]).reset_index(drop=True)
            part.to_parquet(DATA_DIR / f"{name}_{sp}.parquet", index=False)
            stats[f"{name}_{sp}"] = {
                "rows": int(len(part)),
                "label_counts": {str(k): int(v) for k, v in part["label"].value_counts().sort_index().items()},
            }
    stats["top_brands"] = rev["brand"].value_counts().head(15).to_dict()
    stats["router_category_counts"] = rt["category"].value_counts().to_dict()

    # ---- gold sample to hand-label (from TEST split only, so eval is unbiased) ----
    test_c = rt[rt["split"] == "test"]
    gold = test_c.sample(min(args.gold_size, len(test_c)), random_state=7)[["review_id", "brand", "text", "category"]]
    gold = gold.rename(columns={"category": "weak_category"})
    gold["gold_category"] = ""
    gold.to_csv(DATA_DIR / "gold_to_label.csv", index=False)

    (DATA_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
