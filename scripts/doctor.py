"""Preflight checker: tells you in plain words whether this computer is ready (Windows, macOS or Linux).

    python scripts/doctor.py

Exit code 0 = ready (warnings are OK), 1 = something blocking was found.
"""
import importlib
import platform
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
problems, warnings = [], []


def ok(msg):
    print(f"  [ OK ] {msg}")


def warn(msg, fix=""):
    print(f"  [WARN] {msg}" + (f"\n         -> {fix}" if fix else ""))
    warnings.append(msg)


def bad(msg, fix=""):
    print(f"  [FAIL] {msg}" + (f"\n         -> {fix}" if fix else ""))
    problems.append(msg)


print("\n1. Computer")
system, machine = platform.system(), platform.machine()
print(f"       {system} {platform.release()} / {machine}")
if system == "Darwin" and machine != "arm64":
    bad(
        "This looks like an Intel Mac (or Python running under Rosetta).",
        "Current PyTorch has no Intel-Mac build. Use an Apple Silicon Mac, a Windows or a Linux machine, "
        "or open your terminal natively (not 'Open using Rosetta').",
    )
elif system == "Darwin":
    ok("Apple Silicon")

print("\n2. Python")
v = sys.version_info
print(f"       Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
if (v.major, v.minor) < (3, 12):
    bad(
        f"Python {v.major}.{v.minor} is too old for the pinned libraries (need 3.12 or 3.13).",
        "Install Python 3.12 (python.org, winget, brew or apt), delete the .venv folder, and rerun: python scripts/setup_env.py",
    )
elif (v.major, v.minor) > (3, 13):
    warn(f"Python {v.major}.{v.minor} is newer than tested (3.12). It may still work.", "If install fails use Python 3.12.")
else:
    ok("Python version is supported")
if sys.prefix == sys.base_prefix:
    warn("Not inside the project's virtual environment (.venv).", "Use the launcher scripts (they switch to .venv automatically), or activate it - see docs/GUIDE.md.")

print("\n3. Libraries")
need = ["torch", "transformers", "accelerate", "sentencepiece", "sklearn", "pandas", "pyarrow", "onnx",
        "onnxruntime", "fastapi", "uvicorn", "streamlit", "requests", "pytest"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        ok(f"{m} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        missing.append(m)
        bad(f"{m} cannot be imported ({type(e).__name__})")
if missing:
    print("         -> Run:  python scripts/setup_env.py   (or: pip install -r requirements.txt inside .venv)")

print("\n4. Graphics chip (GPU) for training")
try:
    import torch

    if torch.cuda.is_available():
        dev = "cuda"
        ok(f"NVIDIA GPU found: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = "mps"
        ok("Apple GPU (MPS) is available - training will use it automatically")
    else:
        dev = "cpu"
        if system == "Darwin":
            warn(
                "PyTorch cannot see the Apple GPU, so training would use the CPU only (very slow).",
                "Needs macOS 12.3+ on Apple Silicon and the standard 'pip install torch' build.",
            )
        elif system == "Windows":
            warn("No GPU used: on Windows the default pip build of PyTorch is CPU-only. Training will be slow.",
                 "For an NVIDIA GPU install a CUDA build of torch from https://pytorch.org/get-started/locally/")
        else:
            warn("No GPU found - training will use the CPU (slow).",
                 "For an NVIDIA GPU make sure the driver is installed (nvidia-smi works).")
    # a real mini training step on that device (catches broken GPU setups early)
    t = time.time()
    lin = torch.nn.Linear(256, 4).to(dev)
    x = torch.randn(64, 256, device=dev)
    loss = lin(x).pow(2).mean()
    loss.backward()
    if dev == "mps":
        torch.mps.synchronize()
    if torch.isnan(loss).item():
        bad(f"A tiny test calculation on {dev} produced NaN.", "Add --cpu when training.")
    else:
        ok(f"Tiny forward+backward test on {dev} worked ({(time.time() - t) * 1000:.0f} ms)")
except Exception as e:  # noqa: BLE001
    bad(f"Could not test the device: {type(e).__name__}: {e}")

print("\n5. Project files")
for f in ["gatekeeper_train", "gatekeeper_val", "gatekeeper_test", "router_train", "router_val", "router_test"]:
    p = ROOT / "data" / f"{f}.parquet"
    if p.exists():
        ok(f"data/{f}.parquet ({p.stat().st_size / 1e6:.0f} MB)")
    else:
        bad(f"data/{f}.parquet is missing", "Build it once with:  python -m src.build_dataset   (downloads ~3 GB, ~1-2 min to process)")
for m in ("gatekeeper", "router"):
    p = ROOT / "models" / f"{m}_onnx" / "model.onnx"
    print(f"       models/{m}_onnx: " + ("trained + exported" if p.exists() else "not created yet (normal before training)"))

print("\n6. Disk space")
free = shutil.disk_usage(ROOT).free / 1e9
if free < 10:
    bad(f"Only {free:.0f} GB free; need about 10 GB (libraries ~2 GB + model download 0.3 GB + trained models ~2.3 GB + temporary files ~2 GB).", "Free some space.")
else:
    ok(f"{free:.0f} GB free")

print("\n7. Internet (needed once, to download the base model ~0.3 GB from Hugging Face)")
cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--deberta-v3-small"
if cache.exists():
    ok("Base model already downloaded (cached)")
else:
    try:
        urllib.request.urlopen("https://huggingface.co/microsoft/deberta-v3-small/resolve/main/config.json", timeout=15)
        ok("huggingface.co is reachable")
    except Exception as e:  # noqa: BLE001
        bad(
            f"Cannot reach huggingface.co ({type(e).__name__}).",
            "Check your internet/VPN/proxy. Some corporate networks block it - try another network.",
        )

print("\n" + "=" * 60)
if problems:
    print(f"NOT READY: {len(problems)} problem(s) found. Fix the [FAIL] lines above, then rerun this script.")
    print("If you are stuck, run:  python scripts/collect_diagnostics.py   and attach diagnostics.txt to a bug report.")
    sys.exit(1)
print("READY." + (f" ({len(warnings)} warning(s) above - read them.)" if warnings else ""))
