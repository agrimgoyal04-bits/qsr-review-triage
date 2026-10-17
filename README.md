# QSR Review Triage

A fast, two-stage text classifier that reads **customer reviews of quick-service restaurants (fast food)** and answers two questions:

1. **Is this a complaint?** - with a probability.
2. **If so, what is it about?** - wait time, food quality, order accuracy, cleanliness, staff attitude, price/value, or other.

Both stages are [`microsoft/deberta-v3-small`](https://huggingface.co/microsoft/deberta-v3-small) models fine-tuned on public restaurant reviews, exported to ONNX, and served through a small FastAPI service with a Streamlit web UI (single review, or CSV/Excel batch upload).

```json
{
  "decision": "COMPLAINT",
  "complaint_probability": 0.97,
  "category": "wait_time_service_speed",
  "category_confidence": 0.61,
  "category_probabilities": {"wait_time_service_speed": 0.61, "order_accuracy": 0.30, "...": "..."},
  "latency_ms": 73
}
```

> **Status: prototype / proof of concept.** Labels are generated automatically (star ratings and keyword rules), the data is US/Canada-centric Yelp English text, and no human-labelled evaluation exists yet. Read [Limitations](#limitations) before drawing conclusions from the numbers.

## Results

Evaluated on **held-out restaurants** (no restaurant appears in both training and test data), using a random sample of the test split. Training: one Apple-silicon GPU, `train_all.py medium` preset (a subset of the training data), about 1.15 s/step.

| Stage | Task | Train rows | Test rows | Accuracy | Macro-F1 | Majority-class baseline |
|---|---|---|---|---|---|---|
| 1 - Gatekeeper | complaint vs. not (1-2 star vs. 4-5 star) | 30,000 (1 epoch) | 5,000 | **97.4%** (+/-0.4) | 0.973 | 59.2% |
| 2 - Router | 7 issue categories (keyword-derived labels) | 20,000 (2 epochs) | 4,000 | **90.4%** (+/-0.9) | 0.875 | 32.5% |

Router per-category F1: wait_time_service_speed 0.944, other 0.930, food_quality 0.909, staff_attitude 0.856, cleanliness_hygiene 0.843, order_accuracy 0.828, price_value 0.815.
Raw numbers incl. confusion matrices: [`docs/results/`](docs/results/). Intervals are 95% binomial confidence intervals for the sampled test sets.

**How to read these numbers:**
- The gatekeeper's "right answer" is the star rating, so 97.4% measures how well it separates negative from positive reviews - not whether a human would call each one an actionable complaint.
- The router's "right answer" comes from keyword rules (`src/qsr_labels.py`). **90.4% is agreement with those rules, not true accuracy.** A hand-labelled gold sample is the next step (see [Roadmap](#roadmap)).
- Only the "medium" preset has been run; the "full" preset (all data, more epochs) has not.

## How it works

```
 Yelp reviews --filter+label--> data/*.parquet --fine-tune--> 2 x DeBERTa-v3-small --export--> ONNX
 (public dataset)            (restaurant-grouped        (PyTorch: CUDA / Apple MPS / CPU)            |
                              train/val/test split)                                                 v
                                                      Streamlit UI  <---HTTP--->  FastAPI + ONNX Runtime

 one review:   text -> [Stage 1: gatekeeper] -- P(complaint) < threshold --> OK  (stop)
                                  |
                                  +-- >= threshold --> [Stage 2: router] -> category (or "uncertain" if low confidence)
```

The cascade means the second model only runs on flagged reviews. Thresholds are configurable (see [Configuration](#configuration)).

## Quickstart (Windows, macOS, Linux)

Requirements: **Python 3.12 or 3.13**, ~10 GB free disk (more on Linux, where PyTorch's default build bundles several GB of NVIDIA libraries), internet access (PyPI, Hugging Face). A GPU is optional but training on CPU is slow.
Use `python3` (macOS/Linux) or `py` (Windows) if `python` is not found. No virtual-environment activation is needed: the scripts manage `.venv` themselves.

```bash
python scripts/setup_env.py        # 1. creates .venv, installs pinned dependencies, runs a readiness check
                                   #    (Linux without an NVIDIA GPU: see docs/GUIDE.md for a much smaller CPU-only install)
python scripts/build_data.py       # 2. one time: downloads the Yelp reviews (~3 GB) and builds the datasets
python scripts/quick_test.py       # 3. ~5-minute self-test; prints the device used and a training-time estimate
python scripts/train_all.py medium # 4. train both stages, export to ONNX, run tests  (add --cpu to force CPU)
python scripts/run_app.py          # 5. start API + web UI (http://localhost:8501)
```

If something fails, `python scripts/collect_diagnostics.py` writes a `diagnostics.txt` (no secrets) to attach to a bug report.
**New to this?** Read the step-by-step [beginner guide](docs/GUIDE.md): per-OS install notes, how to read the results, troubleshooting, glossary.

### Platform support

| Platform | Accelerator | Status |
|---|---|---|
| Linux x86-64 | CPU | Tested: setup, self-test, training, export, API, UI, test suite |
| macOS, Apple Silicon | Apple GPU (MPS) | Tested: medium-preset training (results above) and self-test |
| Windows 10/11 | CPU or NVIDIA (CUDA) | **Not tested yet.** Launchers are plain Python (Windows paths unit-tested); the default Windows pip wheel of PyTorch is CPU-only - install a CUDA build for an NVIDIA GPU |
| Linux/Windows, NVIDIA GPU | CUDA | **Not tested yet** (standard PyTorch path) |
| Intel Macs | - | Not supported (current PyTorch has no Intel-Mac build) |

Reports for untested platforms are welcome.

## Training options

`python scripts/train_all.py <preset> [flags passed to src/train.py]`

| Preset | What it does | Use for |
|---|---|---|
| `smoke` | a few hundred samples | pipeline check only (useless models) |
| `medium` | gatekeeper 30K x 1 epoch, router 20K x 2 epochs | first real run; produced the results above |
| `full` | all data: gatekeeper 105K x 2, router 64K x 3 | best quality, long |

Useful flags: `--cpu`, `--batch-size 8` (less memory), `--max-length 128` (shorter texts, faster), `--lr 1e-5`, `--pad-multiple 64`.
Existing models are moved to `models/_previous_<time>/`, never overwritten. Run `python -m src.train --help` for everything.

## Configuration

Environment variables (set them in front of the command, e.g. `COMPLAINT_THRESHOLD=0.8 python scripts/run_app.py`; on Windows cmd: `set COMPLAINT_THRESHOLD=0.8 && python scripts/run_app.py`). Nothing reads a `.env` file automatically - `.env.example` just documents the names.

| Variable | Default | Meaning |
|---|---|---|
| `COMPLAINT_THRESHOLD` | 0.5 | probability above which a review is flagged as a complaint |
| `CATEGORY_MIN_CONFIDENCE` | 0.5 | below this the router's answer is reported as `uncertain` |
| `MODELS_DIR` | `./models` | where `gatekeeper_onnx/` and `router_onnx/` live |
| `API_KEY` | unset | if set, API calls must send header `X-API-Key: <value>` |
| `QSR_API_URL` | `http://127.0.0.1:8000` | where the UI finds the API |

### API

```bash
curl -s -X POST localhost:8000/api/v1/classify-review -H 'content-type: application/json' \
  -d '{"text":"Waited 25 min in the drive thru and they forgot my fries","brand":"Wendys"}'
```
Endpoints: `GET /health`, `POST /api/v1/classify-review`, `POST /api/v1/classify-batch` (up to 64 reviews per call). Interactive docs at `/docs`.

**Security:** the API has no authentication unless `API_KEY` is set and listens on `127.0.0.1` only. If you expose it beyond your machine, set `API_KEY`, use HTTPS, and add rate limiting.

## Project layout

```
scripts/        cross-platform launchers (Python): setup_env, build_data, quick_test, train_all, run_app, doctor, collect_diagnostics
src/            config, weak-label rules, dataset builder, training, ONNX export, inference cascade, upload parsing, slowdown detector
api/main.py     FastAPI service
app.py          Streamlit UI (single review + CSV/Excel batch)
tests/          unit + API tests (API tests skip until models are exported; data tests skip until data is built)
docs/           GUIDE.md (beginner guide), results/ (metrics JSON)
data/           sample_reviews.csv/.xlsx (synthetic examples). Built datasets are generated here but git-ignored
models/         trained models are written here (git-ignored)
```

## Data and licensing

- **Code:** MIT (see [LICENSE](LICENSE)).
- **Data:** the [Yelp Open Dataset](https://www.yelp.com/dataset), fetched at build time from Hugging Face mirrors (`yashraizad/yelp-open-dataset-business`, `...-reviews`). It is **not** included in this repository and its own terms apply (the Yelp dataset is licensed for academic / non-commercial use and may not be redistributed) - check them before any commercial use. Generated datasets and trained models are git-ignored for this reason.
- **Selection:** businesses with the "Fast Food" category belonging to a chain with >= 5 locations (5,359 locations, 83 brands, e.g. McDonald's, Chipotle, Taco Bell, Chick-fil-A). Reviews under 20 characters and duplicates removed; 3-star reviews dropped. Split 80/10/10 by restaurant using an md5 hash of the business id.
- `data/sample_reviews.*` are short synthetic examples written for this project.
- **Base model:** `microsoft/deberta-v3-small` (MIT).

## Limitations

1. **Weak labels.** Gatekeeper label = star rating; router label = keyword rules, single-label (reviews often mention several issues). Reported scores are against these labels.
2. **Not global.** Yelp is mostly the US and Canada, English only. Behaviour on other countries, languages, or channels (app stores, social media, call transcripts) is untested.
3. **Public reviews are not internal customer data;** distribution shift to a company's own feedback is likely.
4. Brand is metadata only (the model does not see it); large chains dominate the data.
5. Training used a subset ("medium" preset); no hyper-parameter search; no probability calibration.
6. Latency (~73 ms median per review, both stages, ONNX Runtime) was measured on a laptop CPU only.

## Roadmap

1. Hand-label `data/gold_to_label.csv` (600 held-out complaints, written by the dataset builder) for an honest router score.
2. Better router labels (LLM- or human-labelled, multi-label), and an "actionable complaint" definition for the gatekeeper.
3. Compare against a TF-IDF + logistic-regression baseline; run the `full` preset.
4. Probability calibration and threshold tuning; int8 quantization of the ONNX models.
5. Multilingual / multi-region data (e.g. `microsoft/mdeberta-v3-base`).

## Development

```bash
python -m pytest -q        # inside .venv, or: python scripts/quick_test.py
```
Dependencies are pinned in `requirements.txt` (Python 3.12/3.13; wheels exist for Windows x86-64, macOS arm64 and Linux x86-64 - see the notes in the file). Design notes worth knowing:

- Models load in **fp32 on purpose**: transformers 5 defaults to the checkpoint dtype (fp16 for this model), which broke the weighted loss and slowed CPU training ~19x.
- ONNX export verifies output parity against a **freshly re-loaded** PyTorch model (the in-memory traced model gave a false mismatch); max logit difference is ~1e-7 to 4e-7.
- Batches are padded to a multiple of 64 tokens: on Apple GPUs, a different tensor shape per batch made training slow down from ~1.8 to ~35 s/step; with few distinct shapes it ran at ~1.15 s/step.

## License

MIT - see [LICENSE](LICENSE).
