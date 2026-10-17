import os

import pandas as pd
import requests
import streamlit as st

from src.tabular import SpreadsheetError, read_reviews_table

API_BASE = os.environ.get("QSR_API_URL", "http://127.0.0.1:8000")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}

st.set_page_config(page_title="System One - QSR Review Triage", page_icon="🍔", layout="wide")
st.title("🍔 System One - QSR Review Triage")
st.caption("Fast two-stage classifier: is it a complaint? -> what is it about? (DeBERTa-v3-small, ONNX)")


def call(path, payload):
    r = requests.post(f"{API_BASE}{path}", json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


tab_one, tab_batch = st.tabs(["Single review", "Batch (CSV or Excel)"])

with tab_one:
    with st.form("one"):
        brand = st.text_input("Brand (optional)", placeholder="e.g. McDonald's")
        text = st.text_area("Customer review", height=160, placeholder="Waited 20 minutes at the drive-thru and they forgot my fries...")
        go = st.form_submit_button("Classify", type="primary")
    if go:
        if not text.strip():
            st.warning("Enter a review.")
        else:
            try:
                d = call("/api/v1/classify-review", {"text": text, "brand": brand or None})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Decision", d["decision"])
                c2.metric("Complaint probability", f"{d['complaint_probability'] * 100:.1f}%")
                c3.metric("Category", d["category"] or "-")
                c4.metric("Latency", f"{d['latency_ms']} ms")
                if d["category_probabilities"]:
                    st.bar_chart(pd.Series(d["category_probabilities"]).sort_values())
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at {API_BASE}. Start it: uvicorn api.main:app")
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.status_code} {e.response.text[:200]}")

with tab_batch:
    st.markdown(
        "Upload a **.csv or .xlsx** file with a column named **text** (the review) and optionally **brand**. "
        "For Excel files the first sheet is used."
    )
    up = st.file_uploader("Spreadsheet", type=["csv", "xlsx"])
    if up is not None:
        try:
            df, n_blank = read_reviews_table(up, up.name)
        except SpreadsheetError as e:
            st.error(str(e))
            df = None
        if df is not None:
            st.success(f"Loaded {len(df):,} reviews from '{up.name}'." + (f" ({n_blank} blank rows skipped.)" if n_blank else ""))
            st.dataframe(df.head(5))
            limit = st.number_input("Max rows to classify", 1, 5000, min(len(df), 500))
            if st.button("Run batch", type="primary"):
                df = df.head(int(limit)).copy()
                rows = []
                bar = st.progress(0.0)
                recs = df.to_dict("records")
                for i in range(0, len(recs), 32):
                    chunk = recs[i : i + 32]
                    payload = {"reviews": [{"text": r["text"][:20000], "brand": str(r["brand"]) if "brand" in r and pd.notna(r["brand"]) else None} for r in chunk]}
                    try:
                        res = call("/api/v1/classify-batch", payload)["results"]
                    except requests.exceptions.RequestException as e:
                        st.error(f"Batch failed: {e}")
                        break
                    rows += res
                    bar.progress(min((i + 32) / len(recs), 1.0))
                out = pd.concat([df.reset_index(drop=True).iloc[: len(rows)], pd.DataFrame(rows).drop(columns=["brand", "category_probabilities"], errors="ignore")], axis=1)
                st.session_state["batch_out"] = out

    out = st.session_state.get("batch_out")
    if out is not None and len(out):
        m1, m2 = st.columns(2)
        m1.metric("Reviews", len(out))
        m2.metric("Complaints", int((out["decision"] == "COMPLAINT").sum()))
        comp = out[out["decision"] == "COMPLAINT"]
        if len(comp):
            st.subheader("Complaints by category")
            st.bar_chart(comp["category"].value_counts())
            if "brand" in comp.columns:
                st.subheader("Complaint count by brand x category")
                st.dataframe(pd.crosstab(comp["brand"], comp["category"]))
        st.dataframe(out)
        st.download_button("Download results CSV", out.to_csv(index=False), "qsr_results.csv", "text/csv")
