"""Build the training datasets (one time). Downloads the Yelp Open Dataset review file (~3 GB) from Hugging Face.

    python scripts/build_data.py                       # full fast-food chain set (~1-2 minutes after download)
    python scripts/build_data.py --max-reviews 20000   # quick dev subset
    python scripts/build_data.py --category-match Pizza --min-locations 10

All arguments are passed to  python -m src.build_dataset .
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.ensure_venv()

if __name__ == "__main__":
    rc = C.run_logged(C.python_cmd("-m", "src.build_dataset", *sys.argv[1:]), C.LOGS / "build_data.log")
    if rc != 0:
        print("\nBuilding the dataset failed. Check your internet connection (needs huggingface.co) and free disk space (~4 GB).")
    sys.exit(rc)
