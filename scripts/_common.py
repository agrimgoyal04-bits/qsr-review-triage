"""Shared helpers for the launcher scripts. Standard library only; works on Windows, macOS and Linux.

The OS-specific decisions live in small pure functions (venv_python, keep_awake_prefix, ...)
so they can be unit-tested on any machine (see tests/test_scripts.py).
"""
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
SUPPORTED_PYTHONS = ((3, 12), (3, 13))  # numpy 2.5 (pinned) needs >= 3.12


# --------------------------------------------------------------------------- environment
def is_supported_python(version) -> bool:
    return tuple(version[:2]) in SUPPORTED_PYTHONS


def venv_python(root: Path = ROOT, windows: Optional[bool] = None) -> Path:
    """Path of the python executable inside <root>/.venv for the given OS."""
    windows = (os.name == "nt") if windows is None else windows
    return root / ".venv" / ("Scripts" if windows else "bin") / ("python.exe" if windows else "python")


def running_in_project_venv(root: Path = ROOT) -> bool:
    try:
        return Path(sys.prefix).resolve() == (root / ".venv").resolve()
    except OSError:
        return False


def ensure_venv() -> None:
    """If not already inside the project's .venv, re-run this same script with the venv's python."""
    vp = venv_python()
    if not vp.exists():
        sys.exit("The virtual environment (.venv) does not exist yet.\n"
                 "Run this first:   python scripts/setup_env.py")
    if running_in_project_venv():
        return
    argv = [str(vp), *sys.argv]
    if os.name != "nt":
        # Replace this process (no wrapper), so Ctrl+C reaches the real program and its cleanup code runs.
        os.execv(str(vp), argv)
        return  # never reached with the real execv
    # Windows: exec is unreliable, so wait for the child. Windows sends Ctrl+C to the child as well,
    # so the wrapper must NOT kill it - just keep waiting while it shuts down.
    proc = subprocess.Popen(argv)
    while True:
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:
            continue


def python_cmd(*args: str) -> List[str]:
    return [str(venv_python()), *args]


# --------------------------------------------------------------------------- keep the computer awake
def keep_awake_prefix(sys_platform: str, which: Callable = shutil.which,
                      probe: Callable[[List[str]], bool] = None) -> List[str]:
    """Command prefix that stops the computer sleeping while a long job runs ([] if not available).

    Windows is handled separately (windows_keep_awake) because it needs an API call, not a wrapper.
    """
    if sys_platform == "darwin" and which("caffeinate"):
        return ["caffeinate", "-i"]
    if sys_platform.startswith("linux") and which("systemd-inhibit"):
        prefix = ["systemd-inhibit", "--what=idle:sleep", "--why=model training"]
        probe = probe or (lambda p: subprocess.run(p + ["true"], capture_output=True).returncode == 0)
        try:
            return prefix if probe(prefix) else []  # e.g. unavailable inside WSL / containers
        except OSError:
            return []
    return []


def windows_keep_awake() -> bool:
    """Ask Windows not to sleep while this launcher process is alive. No-op elsewhere."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- running + logging
def run_logged(cmd: Sequence[str], log_path: Optional[Path] = None, echo: bool = True,
               env_extra: Optional[dict] = None, append: bool = True) -> int:
    """Run a command, optionally echoing output live AND saving it to a log file. Returns the exit code.

    Raw bytes are passed through so progress bars (which rewrite the line with \\r) still look right.
    """
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if env_extra:
        env.update(env_extra)
    log = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab" if append else "wb")
    try:
        proc = subprocess.Popen(list(cmd), cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                if echo:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                if log:
                    log.write(chunk)
                    log.flush()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        return proc.wait()
    finally:
        if log:
            log.close()


def say(msg: str = "", log_path: Optional[Path] = None) -> None:
    """Print (ASCII-safe) and optionally append to a log file."""
    print(msg, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- training presets + estimates
# Extra command-line flags given after the preset are appended last, so they override these (argparse: last wins).
PRESETS = {
    "smoke": {"gatekeeper": ["--max-train", "160", "--max-eval", "64", "--epochs", "1", "--max-length", "128"],
              "router": ["--max-train", "160", "--max-eval", "64", "--epochs", "1", "--max-length", "128",
                         "--class-weights"]},  # pipeline check only: the resulting models are useless
    "medium": {"gatekeeper": ["--max-train", "30000", "--max-eval", "5000", "--epochs", "1"],
               "router": ["--max-train", "20000", "--max-eval", "4000", "--epochs", "2", "--class-weights"]},
    "full": {"gatekeeper": ["--epochs", "2"],
             "router": ["--epochs", "3", "--class-weights"]},
}


def preset_args(name: str, stage: str, extra: Sequence[str] = ()) -> List[str]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Choose one of: {', '.join(PRESETS)}")
    return [*PRESETS[name][stage], *extra]


def parse_training_log(text: str):
    """Return (device or None, [train samples/second for each training run])."""
    dev = re.search(r"Training device: (\w+)", text)
    speeds = [float(x) for x in re.findall(r"'train_samples_per_second': '([\d.]+)'", text)]
    return (dev.group(1) if dev else None), speeds


def estimate_hours(sample_passes: int, samples_per_second: float) -> float:
    return sample_passes / samples_per_second / 3600.0


# sample-passes = (training rows x epochs) summed over both stages
SAMPLE_PASSES = {"medium": 30000 * 1 + 20000 * 2, "full": 104854 * 2 + 63757 * 3}
