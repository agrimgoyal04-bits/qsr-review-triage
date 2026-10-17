"""Train both stages, export to ONNX, run the tests - with logs.

    python scripts/train_all.py medium            # recommended first real run
    python scripts/train_all.py full              # all data, long
    python scripts/train_all.py smoke             # a few minutes, pipeline check only (models are useless)
    python scripts/train_all.py medium --cpu      # extra flags are passed to src/train.py (e.g. --batch-size 8)

Previously trained models in models/ are moved to models/_previous_<time>/ first, never overwritten.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.ensure_venv()


def main() -> int:
    args = sys.argv[1:]
    preset = args[0] if args and not args[0].startswith("-") else "medium"
    extra = args[1:] if args and not args[0].startswith("-") else args
    if preset not in C.PRESETS:
        print(f"Unknown preset '{preset}'. Use one of: {', '.join(C.PRESETS)}")
        return 1

    log = C.LOGS / f"train_{preset}_{C.timestamp()}.log"
    C.LOGS.mkdir(exist_ok=True)

    # keep any earlier trained models
    models = C.ROOT / "models"
    old = [models / n for n in ("gatekeeper_standard", "router_standard", "gatekeeper_onnx", "router_onnx") if (models / n).exists()]
    if old:
        dest = models / f"_previous_{C.timestamp()}"
        dest.mkdir()
        for p in old:
            shutil.move(str(p), str(dest / p.name))
        C.say(f"Existing models were moved to {dest.relative_to(C.ROOT)}", log)

    prefix = C.keep_awake_prefix(sys.platform)
    if C.windows_keep_awake():
        prefix = []
    C.say(f"Preset: {preset}    Log file: {log.relative_to(C.ROOT)}", log)
    C.say("Keep this window open and the computer plugged in (the script asks the OS not to sleep). Ctrl+C stops it.", log)
    C.say("The progress bar shows time remaining, e.g. 1:12:30<2:40:10 = elapsed<remaining.\n", log)

    def step(title: str, *cmd: str, wrap: bool = False) -> bool:
        C.say(f"== {title} ==", log)
        full = (prefix if wrap else []) + C.python_cmd(*cmd)
        return C.run_logged(full, log) == 0

    ok = (
        step("STAGE 1: gatekeeper", "-m", "src.train", "--stage", "gatekeeper", *C.preset_args(preset, "gatekeeper", extra), wrap=True)
        and step("STAGE 2: router", "-m", "src.train", "--stage", "router", *C.preset_args(preset, "router", extra), wrap=True)
        and step("EXPORT to ONNX", "-m", "src.export_onnx", "--stage", "both")
        and step("TESTS", "-m", "pytest", "-q")
    )
    if not ok:
        C.say("\n*** FAILED (see the last lines above). ***", log)
        C.say(f"Log: {log}", log)
        C.say("Tips: GPU problems -> add --cpu.  Out of memory -> add --batch-size 8.  Much slower than expected -> docs/GUIDE.md.", log)
        C.say("Still stuck? Run  python scripts/collect_diagnostics.py  and attach diagnostics.txt to a bug report.", log)
        return 1

    bar = "=" * 62
    C.say("\n" + bar + "\nTRAINING FINISHED. Results on held-out test restaurants:", log)
    for s in ("gatekeeper", "router"):
        m = json.loads((models / f"{s}_standard" / "metrics.json").read_text())
        C.say(f"  {s:11s} accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}  (trained on {m['n_train']} reviews)", log)
    C.say("Details (per class, confusion matrix): models/<stage>_standard/metrics.json", log)
    C.say("NOTE: labels are automatic (stars / keywords); the router score is optimistic. See README 'Limitations'.", log)
    C.say("Next step:   python scripts/run_app.py\n" + bar, log)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Nothing was saved for the run in progress.")
