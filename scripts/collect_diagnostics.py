"""Collect NON-SECRET information about your setup into diagnostics.txt (for bug reports).

    python scripts/collect_diagnostics.py

Includes: OS/Python/library versions, the readiness-check output, folder listing, last lines of recent logs.
Does NOT include: passwords, environment variable values, or your data.
"""
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

KEY_PACKAGES = ("torch", "transformers", "accelerate", "numpy", "pandas", "onnx", "onnxruntime",
                "scikit-learn", "sentencepiece", "streamlit", "fastapi", "tokenizers", "openpyxl")
ENV_PREFIXES = ("PYTORCH", "HF_", "TOKENIZERS", "MODELS_DIR", "COMPLAINT", "CATEGORY", "CUDA")


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(C.ROOT))
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(could not run {cmd[0]}: {type(e).__name__})"


def main():
    out = []
    add = out.append
    add("=== system ===")
    add(f"{platform.platform()} | machine={platform.machine()} | python={sys.version.split()[0]} ({sys.executable})")
    add(f"cpu_count={os.cpu_count()}")
    if platform.system() == "Darwin":
        add(sh(["sysctl", "-n", "machdep.cpu.brand_string"]))
        add("RAM bytes: " + sh(["sysctl", "-n", "hw.memsize"]))
    elif platform.system() == "Linux":
        add(sh(["sh", "-c", "grep -m1 MemTotal /proc/meminfo"]))
    elif platform.system() == "Windows":
        add(sh(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"]))
    add(sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]) if shutil_which("nvidia-smi") else "no nvidia-smi")

    vp = C.venv_python()
    if vp.exists():
        add("\n=== venv ===")
        add(sh([str(vp), "--version"]))
        pk = sh([str(vp), "-m", "pip", "list"]).splitlines()
        add("\n".join(l for l in pk if l.split(" ")[0].lower() in KEY_PACKAGES))
        add("\n=== doctor ===")
        add(sh([str(vp), "scripts/doctor.py"]))
    else:
        add("\nNO .venv FOUND (setup not done)")

    add("\n=== environment variable NAMES (values hidden) ===")
    add("\n".join(f"{k}=<hidden>" for k in sorted(os.environ) if k.upper().startswith(ENV_PREFIXES)) or "(none)")
    add("\n=== folders ===")
    for d in ("data", "models"):
        p = C.ROOT / d
        add(f"{d}/: " + (", ".join(sorted(x.name for x in p.iterdir())) if p.exists() else "missing"))
    logs = sorted(C.LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:3] if C.LOGS.exists() else []
    for lg in logs:
        add(f"\n=== last 60 lines of {lg.name} ===")
        add("\n".join(lg.read_text(errors="replace").splitlines()[-60:]))

    dest = C.ROOT / "diagnostics.txt"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"Created {dest}  (contains no passwords). Attach it to your bug report.")


def shutil_which(name):
    import shutil

    return shutil.which(name)


if __name__ == "__main__":
    main()
