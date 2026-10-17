# Beginner's guide

A step-by-step walkthrough for people who have not trained a model before. It works on Windows, macOS and Linux; where the steps differ, each system is spelled out.

Contents: [What this does](#1-what-this-does) - [What you need](#2-what-you-need) - [Step by step](#3-step-by-step) - [Reading the results](#4-reading-the-results) - [Troubleshooting](#5-troubleshooting) - [Glossary](#6-glossary)

---

## 1. What this does

You give it a customer review of a fast-food restaurant. It tells you:

1. **Is it a complaint?** (a probability, e.g. 97%)
2. **If yes, what about?** wait time, food quality, wrong/missing items, cleanliness, rude staff, price, or other.

### How a review flows through it

```
  "Waited 25 min in the drive thru and they forgot my fries"
                          |
                          v
        +-------------------------------------+
        | TOKENIZER: cut text into word pieces|
        +-------------------------------------+
                          |
                          v
        +-------------------------------------+
        | STAGE 1  GATEKEEPER (AI model)      |    outputs: P(complaint) = 0.97
        | "Is this a complaint?"              |
        +-------------------------------------+
                          |
            +-------------+--------------+
            |                            |
   P < threshold (0.5)           P >= threshold
            |                            |
            v                            v
   decision: OK                +-------------------------------------+
   (stop - saves time)         | STAGE 2  ROUTER (AI model)          |
                               | "What is it about?"                 |
                               | wait_time        0.61               |
                               | order_accuracy   0.30  ... etc.     |
                               +-------------------------------------+
                                             |
                       best guess below 50% confident? --yes--> "uncertain"
                                             |no
                                             v
                          decision: COMPLAINT, category: wait_time_service_speed
```

### How the pieces fit together

```
  STEP A: prepare data          STEP B: train (you)                      STEP C: use it
+---------------------------+  +-----------------------------------+   +--------------------------+
| Yelp reviews (3 GB)       |  | base model "DeBERTa-v3-small"      |   | FastAPI service          |
|   filter fast-food chains |  | (already understands English)      |   |   ^                      |
|   label automatically     |->|   + our labeled reviews            |-->| Streamlit web page       |
|   split by restaurant     |  |   = 2 specialised models           |   |  - type one review       |
| -> data/*.parquet         |  |   -> export to ONNX (fast format)  |   |  - or upload CSV / Excel |
+---------------------------+  +-----------------------------------+   +--------------------------+
 scripts/build_data.py          scripts/setup_env.py, quick_test.py,     scripts/run_app.py
                                train_all.py
```

### Where do the "right answers" come from?

Nobody labelled these reviews by hand. They are generated automatically:

| Model | Right answer comes from | Weakness |
|---|---|---|
| Gatekeeper | Star rating: 1-2 stars = complaint, 4-5 stars = not (3 stars dropped) | It is basically a negative-vs-positive detector. |
| Router | Keyword rules in `src/qsr_labels.py` ("cold", "stale" -> food quality; "rude" -> staff ...) | The model largely learns the keywords, so its scores look better than the real quality. |

Keep this in mind whenever you read a score.

---

## 2. What you need

- **Python 3.12 or 3.13** (older versions do not work with the pinned libraries).
- About **10 GB free disk space** (a bit more on Linux), and an internet connection.
- Optional but strongly recommended: a **GPU** - an Apple Silicon Mac or an NVIDIA graphics card. On a plain CPU, training is many times slower.
- Time: setup + data ~20-30 minutes; the `medium` training run took about 1.5 hours on an Apple-silicon GPU (it will be much longer on a CPU - the self-test prints an estimate).

What gets downloaded:

| What | Size | When | From |
|---|---|---|---|
| Python libraries (PyTorch, transformers, ...) | ~1-2 GB (macOS, Windows); several GB on Linux (NVIDIA libraries) | `setup_env.py` | pypi.org |
| Yelp review file (all business types) | ~3 GB | `build_data.py` (once) | huggingface.co |
| Base model `microsoft/deberta-v3-small` | ~0.3 GB | first training run | huggingface.co |

No accounts or logins are needed. After these downloads everything runs offline.

**Linux without an NVIDIA GPU:** the default PyTorch download includes several GB of GPU libraries you will not use. For a much smaller CPU-only install, run setup like this:

```bash
PIP_INDEX_URL=https://download.pytorch.org/whl/cpu PIP_EXTRA_INDEX_URL=https://pypi.org/simple python scripts/setup_env.py
```

### Installing Python 3.12

| System | How |
|---|---|
| Windows | `winget install Python.Python.3.12`, or the installer from python.org (tick **"Add python.exe to PATH"**) |
| macOS | installer from python.org (then double-click `Install Certificates.command` once), or `brew install python@3.12` |
| Linux | `sudo apt install python3.12 python3.12-venv` (Debian/Ubuntu), or your distro's equivalent |

Afterwards **close and reopen your terminal**. The scripts search for 3.12/3.13 themselves, so the "default" python may stay older.

Which command starts Python? Try `python --version`; if that fails try `python3 --version` (macOS/Linux) or `py --version` (Windows). Use whichever works wherever this guide says `python`.

---

## 3. Step by step

Open a terminal (Windows: PowerShell or Terminal; macOS: Terminal; Linux: any) and go into the project folder, e.g. `cd path/to/qsr-review-triage`.
Tip: type `cd ` and **drag the folder into the terminal window** to paste its path.

### Step 1 - Setup (once)
```bash
python scripts/setup_env.py
```
Creates a private environment in a `.venv` folder (nothing is installed system-wide), installs the libraries (3-10 minutes), and runs a readiness check. **Success:** it ends with `READY.`. A `[WARN]` is a heads-up; a `[FAIL]` explains what to fix.

### Step 2 - Build the datasets (once)
```bash
python scripts/build_data.py
```
Downloads the Yelp review file (~3 GB) and filters it down to fast-food chain reviews (about 1-2 minutes of processing). **Success:** it prints dataset statistics and creates `data/*.parquet`.

### Step 3 - Self-test (5 minutes)
```bash
python scripts/quick_test.py
```
Trains two *tiny* models (results are meaningless on purpose), converts them, runs all tests, and measures your speed. **Success:**
```
SELF-TEST PASSED - your setup works.
Training device used: MPS            <- or CUDA; "CPU" means no GPU is being used
Measured speed (tiny run, rough): ... samples/second
  medium preset: about ... hours      <- real runs can be up to ~2x slower than this estimate
```
If the device says `CPU` and you expected a GPU, see [GPU problems](#gpu-problems).

### Step 4 - Real training
```bash
python scripts/train_all.py medium
```
`medium` trains on a part of the data and is enough for real, meaningful numbers. Use `full` for everything (long). Watch for:
- `>>> Training device: MPS` (or CUDA) at the start.
- A progress bar with time remaining, like `45%|####5 | 1200/2600 [12:30<14:10, 1.5it/s]` - after the `<` is the time left. Steps should stay about the same speed throughout.
- Every so often a line with `eval_accuracy` / `eval_macro_f1` (scores on validation data).
- Order: stage 1, stage 2, export to ONNX, tests. Everything is saved to `logs/train_<preset>_<time>.log`.

Keep the window open and the computer plugged in (the script asks the OS not to sleep). `Ctrl+C` stops it; an interrupted run saves nothing. Earlier trained models are moved to `models/_previous_<time>/`, never deleted.

Add flags at the end if needed, for example: `python scripts/train_all.py medium --cpu`, `--batch-size 8`, `--max-length 128`.

### Step 5 - Use it
```bash
python scripts/run_app.py
```
Opens `http://localhost:8501`.
- **Single review** tab: paste a review, click *Classify*.
- **Batch (CSV or Excel)** tab: upload a `.csv` or `.xlsx` with a column named `text` (optionally `brand`; Excel uses the first sheet), click *Run batch*, view totals by category and brand, download the results. Example files: `data/sample_reviews.xlsx`, `data/sample_reviews.csv`. Old `.xls` files are not supported - re-save as `.xlsx`.
- `Ctrl+C` in the terminal stops everything.

Next time, only `python scripts/run_app.py` is needed - trained models stay in `models/`.

---

## 4. Reading the results

After training, `models/gatekeeper_standard/metrics.json` and `models/router_standard/metrics.json` hold the scores on the **test** group (restaurants never seen in training).

| Term | Plain meaning |
|---|---|
| **Accuracy** | Out of 100 reviews, how many it got right. |
| **Precision** (for "complaint") | When it says "complaint", how often is it truly one? |
| **Recall** (for "complaint") | Of all real complaints, how many did it catch? |
| **F1** | One number combining precision and recall (0 = awful, 1 = perfect). |
| **Macro-F1** | Average F1 across categories, treating small ones equally - better than accuracy for uneven classes. |
| **Confusion matrix** | Grid: rows = true category, columns = what the model said. A strong diagonal is good. |

A score only means something next to a dumb baseline: always guessing the biggest class gets ~59% on the gatekeeper test set and ~33% on the router. Remember the labels are automatic (section 1), so treat the router score as "agreement with the keyword rules".

Two knobs: `COMPLAINT_THRESHOLD` (raise to flag fewer, surer complaints) and `CATEGORY_MIN_CONFIDENCE` (below it, category = "uncertain"). Set them in front of the command:

| System | Example |
|---|---|
| macOS / Linux | `COMPLAINT_THRESHOLD=0.8 python scripts/run_app.py` |
| Windows cmd | `set COMPLAINT_THRESHOLD=0.8 && python scripts/run_app.py` |
| Windows PowerShell | `$env:COMPLAINT_THRESHOLD=0.8; python scripts/run_app.py` |

---

## 5. Troubleshooting

**To ask for help:** run `python scripts/collect_diagnostics.py`. It writes `diagnostics.txt` (system info, library versions, readiness check, last log lines - no passwords). Attach it plus the last ~30 lines of terminal output to your bug report.

| What you see | Meaning | What to do |
|---|---|---|
| `python: command not found`, or an old version | Python missing/wrong | Install 3.12 (section 2), reopen the terminal; try `python3` / `py` |
| `Python 3.12 or 3.13 was not found` | Only older Python installed | Install 3.12; the setup script finds it automatically |
| `xcrun: error: invalid active developer path` (macOS) | Developer tools missing | `xcode-select --install` |
| `Intel Macs are not supported` | Intel Mac | Use an Apple Silicon, Windows or Linux machine. On Apple Silicon: terminal > Get Info > untick *Open using Rosetta* |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Missing certificates, or a company network inspecting traffic | macOS python.org install: run `Install Certificates.command`. Else try another network |
| Download stalls / `Connection timed out` / `Could not resolve host` / `huggingface.co [FAIL]` | No internet, VPN, proxy or firewall | Disconnect VPN, try another network, rerun (downloads resume) |
| `No matching distribution found` / `requires a different Python` | Wrong Python version | Needs 3.12 or 3.13. Delete the `.venv` folder and rerun `setup_env.py` |
| `python3-venv` / `ensurepip` error (Debian/Ubuntu) | venv module not installed | `sudo apt install python3.12-venv` |
| `data/...parquet is missing` | Datasets not built | `python scripts/build_data.py` |
| Self-test says device `CPU` | No GPU in use | See [GPU problems](#gpu-problems) |
| `NotImplementedError: operator ... not implemented for the MPS device` | An Apple-GPU op is unsupported | Normally avoided automatically (CPU fallback is on). If it stops: add `--cpu` |
| `Loss became NaN/inf` | GPU produced broken numbers | Add `--cpu` or `--lr 1e-5` |
| `out of memory`, `Killed`, or the machine becomes very slow | Not enough memory for the batch | Add `--batch-size 8` (or 4) and/or `--max-length 128`; close other apps |
| `CUDA out of memory` | GPU memory full | Same: smaller `--batch-size` |
| Training stopped because the computer slept / terminal closed | Interrupted | Rerun the same command (it restarts). Stay plugged in, window open |
| `models/gatekeeper_onnx/model.onnx not found` | Not trained yet | `python scripts/train_all.py medium` |
| `ONNX output deviates from PyTorch ... do not deploy` | Export safety check failed | Do not use that export; report it with diagnostics |
| `something is already using port 8000` | An earlier copy is still running | Close that terminal (or end the python process) and retry |
| UI says "Cannot reach API" | API not running | Start with `run_app.py` (starts both) |
| Tests show `skipped` | Data/models not present yet | Normal before building data / training |
| `No space left on device` | Disk full | Free ~10 GB; delete `logs/` and `models/_previous_*` |
| Upload: "No review-text column found" | Column is not called `text` | Rename the header to `text` (also accepted: review, review_text, comment, feedback) |
| Upload: "Could not read..." / "Unsupported file type" | Old `.xls`, password-protected, or not a spreadsheet | Re-save as `.xlsx` or CSV UTF-8, without a password |
| Windows: `running scripts is disabled` | PowerShell policy (only for `.ps1` files) | Not an issue here - run `python scripts/...`, not `.ps1` |

### GPU problems

Run `python scripts/doctor.py` and read section 4 of its output.

- **Apple Silicon:** needs macOS 12.3+ and the standard `torch` wheel (installed by `setup_env.py`). Make sure the terminal is not running under Rosetta.
- **NVIDIA (Windows):** the default pip wheel of PyTorch on Windows is **CPU-only**. Install a CUDA build from <https://pytorch.org/get-started/locally/> inside `.venv`, then rerun the self-test.
- **NVIDIA (Linux):** install the NVIDIA driver (`nvidia-smi` must work); the default Linux wheel includes CUDA.
- No GPU: you can still train with `--cpu`, but expect hours-to-days; use `medium` or `smoke`.

### Training starts fast, then slows down

Seen on Apple GPUs: ~1.8 s/step at first, then 10-40 s/step. Cause: a different tensor size for every batch makes the GPU backend cache more and more until memory runs out and the system swaps to disk. Current versions pad batches to multiples of 64 tokens (`--pad-multiple`), free cached GPU memory every 25 steps, print `[mem]` lines every 100 steps, and print `SLOWDOWN DETECTED` if steps become 4x slower than at the start. With these, steps stayed at ~1.15 s on an Apple GPU.
If it still slows down: `Ctrl+C`, retry with `--batch-size 8`, then `--max-length 128`; report the `[mem]` lines, the machine's RAM and swap usage.

---

## 6. Glossary

- **Classifier** - a program that puts an input (text) into one of several categories.
- **Model** - the trained "brain": a big file of numbers that turns text into probabilities.
- **Fine-tuning** - taking a pre-trained model and training it a bit more on your own labeled examples.
- **DeBERTa-v3-small** - the pre-trained English model used here (Microsoft, MIT licence).
- **LLM** - a large generative language model (ChatGPT-style). This project does *not* use one; it uses a much smaller model that only classifies.
- **Token** - a word piece the model reads ("waited" -> "wait" + "ed").
- **Probability** - a 0-1 (0-100%) confidence value.
- **Epoch** - one full pass through the training data. **Batch** - how many reviews are processed at once.
- **Learning rate** (`--lr`) - the size of each adjustment step during training. **Loss** - the training error (should go down; NaN = broken).
- **GPU / CUDA / MPS / CPU** - the graphics chip / NVIDIA's and Apple's ways for PyTorch to use it / the general processor (works, but slow for training).
- **PyTorch** - the library that trains the model. **ONNX** - a standard model file format that runs fast and needs no PyTorch.
- **FastAPI / API** - the service other programs send reviews to over HTTP. **Streamlit** - the library behind the web page.
- **Virtual environment (`.venv`)** - a private folder of libraries for this project only.
- **Hugging Face** - a website hosting free AI models and datasets.
- **Weak labels** - answers generated automatically (stars, keywords) instead of by humans: cheap but noisy.
- **Train / validation / test** - data for learning / for choosing the best version / for the final honest score.
