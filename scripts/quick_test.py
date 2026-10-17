"""Self-test (a few minutes): trains two TINY models, exports them, runs the tests, and estimates how long
real training will take on THIS computer. Nothing it creates is kept.

    python scripts/quick_test.py [--cpu]

Needs the dataset (data/*.parquet). If missing:  python -m src.build_dataset
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.ensure_venv()

LOG = C.LOGS / "quick_test.log"
Q = C.ROOT / "models" / "_quick"


def fail(step: str):
    C.say(f"\n*** SELF-TEST FAILED at step: {step} ***", LOG)
    C.say(f"See the error above. Log file: {LOG}", LOG)
    C.say("To get help run:  python scripts/collect_diagnostics.py  and attach diagnostics.txt to a bug report.", LOG)
    shutil.rmtree(Q, ignore_errors=True)
    sys.exit(1)


def run(step: str, *args: str):
    if C.run_logged(C.python_cmd(*args), LOG, env_extra=ENV) != 0:
        fail(step)


def main():
    C.LOGS.mkdir(exist_ok=True)
    LOG.write_text("")
    extra = [a for a in sys.argv[1:]]
    shutil.rmtree(Q, ignore_errors=True)
    Q.mkdir(parents=True)

    C.say("== Step 1/5: readiness check ==", LOG)
    run("readiness check", "scripts/doctor.py")

    C.say("\n== Step 2/5: tiny training, stage 1 (gatekeeper) ==", LOG)
    run("gatekeeper training", "-m", "src.train", "--stage", "gatekeeper", "--max-train", "320", "--max-eval", "64",
        "--epochs", "1", "--max-length", "128", "--batch-size", "16", "--out", str(Q / "gatekeeper_standard"), *extra)

    C.say("\n== Step 3/5: tiny training, stage 2 (router) ==", LOG)
    run("router training", "-m", "src.train", "--stage", "router", "--class-weights", "--max-train", "160",
        "--max-eval", "64", "--epochs", "1", "--max-length", "128", "--batch-size", "16",
        "--out", str(Q / "router_standard"), *extra)

    C.say("\n== Step 4/5: export to ONNX (fast serving format) and verify ==", LOG)
    for s in ("gatekeeper", "router"):
        run(f"export {s}", "-m", "src.export_onnx", "--src", str(Q / f"{s}_standard"), "--dst", str(Q / f"{s}_onnx"))

    C.say("\n== Step 5/5: automated tests ==", LOG)
    run("tests", "-m", "pytest", "-q")

    dev, speeds = C.parse_training_log(LOG.read_text(errors="replace"))
    bar = "=" * 62
    C.say("\n" + bar, LOG)
    C.say("SELF-TEST PASSED - your setup works.", LOG)
    C.say(f"Training device used: {dev or 'unknown'}", LOG)
    if dev and dev.upper() == "CPU":
        C.say("!! It used the CPU, not a GPU. Real training will be very slow (see docs/GUIDE.md, 'GPU problems').", LOG)
    if speeds:
        s = speeds[0]
        C.say(f"Measured speed (tiny run, rough): {s:.1f} samples/second", LOG)
        C.say("Rough time estimates for real runs (real texts are longer: expect up to ~2x slower than this):", LOG)
        for name in ("medium", "full"):
            C.say(f"  {name} preset: about {C.estimate_hours(C.SAMPLE_PASSES[name], s):.1f} hours", LOG)
    C.say("Next step:   python scripts/train_all.py medium", LOG)
    C.say(bar, LOG)
    shutil.rmtree(Q, ignore_errors=True)


ENV = {"MODELS_DIR": str(Q)}  # makes tests/test_api.py exercise the tiny exported models

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        shutil.rmtree(Q, ignore_errors=True)
        sys.exit("\nInterrupted.")
