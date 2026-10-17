"""One-time setup: finds a suitable Python (3.12 or 3.13), creates .venv, installs the libraries, runs the checker.

    python scripts/setup_env.py

Works on Windows, macOS and Linux. Run it with any Python 3; it will look for 3.12/3.13 itself.
"""
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

LOG = C.LOGS / "setup.log"
PIP_LOG = C.LOGS / "setup_pip.log"  # raw pip output, kept apart so error excerpts show only pip messages


def candidate_commands(os_name: str) -> List[List[str]]:
    """Interpreter commands to try, best first. os_name is os.name ('nt' or 'posix')."""
    cands = [[sys.executable]]
    cands += [[n] for n in ("python3.13", "python3.12", "python3", "python")]
    if os_name == "nt":
        cands += [["py", "-3.13"], ["py", "-3.12"]]  # Windows "py" launcher
    return cands


def probe(cmd: List[str]) -> Optional[tuple]:
    try:
        out = subprocess.run([*cmd, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        major, minor = out.stdout.strip().split(".")
        return int(major), int(minor)
    except ValueError:
        return None


def find_python() -> Optional[List[str]]:
    import os

    for cmd in candidate_commands(os.name):
        ver = probe(cmd)
        if ver and C.is_supported_python(ver):
            return cmd
    return None


INSTALL_HELP = """
ERROR: Python 3.12 or 3.13 was not found on this computer.

Install one (pick what fits your system), then CLOSE and REOPEN your terminal and run this script again:
  * Any OS      : download from https://www.python.org/downloads/  (choose 3.12.x)
  * Windows     : winget install Python.Python.3.12
  * macOS       : brew install python@3.12
  * Ubuntu/Debian: sudo apt install python3.12 python3.12-venv
"""


def main() -> int:
    C.LOGS.mkdir(exist_ok=True)
    LOG.write_text("")
    C.say("== QSR review triage: setup ==", LOG)

    if platform.system() == "Darwin" and platform.machine() != "arm64":
        C.say("\nERROR: Intel Macs are not supported (current PyTorch has no Intel-Mac build).", LOG)
        C.say("If this IS an Apple Silicon Mac, your terminal may run under Rosetta: Terminal > Get Info > untick 'Open using Rosetta'.", LOG)
        return 1

    py = find_python()
    if py is None:
        C.say(f"Your default python is {sys.version.split()[0]}." + INSTALL_HELP, LOG)
        return 1
    C.say(f"Using Python: {' '.join(py)} ({subprocess.run([*py, '--version'], capture_output=True, text=True).stdout.strip()})", LOG)

    vp = C.venv_python()
    if vp.exists():
        ver = probe([str(vp)])
        if not (ver and C.is_supported_python(ver)):
            C.say("The existing .venv is unusable. Delete the .venv folder and run this script again.", LOG)
            return 1
        C.say(".venv already exists - reusing it.", LOG)
    else:
        C.say("Creating virtual environment (.venv) ...", LOG)
        if C.run_logged([*py, "-m", "venv", str(C.ROOT / ".venv")], LOG, echo=False) != 0:
            C.say(f"ERROR: could not create the virtual environment (see {LOG}).", LOG)
            C.say("On Ubuntu/Debian you may need:  sudo apt install python3-venv", LOG)
            return 1

    C.say("Upgrading pip ...", LOG)
    C.run_logged([str(vp), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], PIP_LOG, echo=False, append=False)

    C.say("Installing libraries (downloads 1-6 GB depending on your OS, can take 5-20 minutes, please wait) ...", LOG)
    rc = C.run_logged([str(vp), "-m", "pip", "install", "-r", "requirements.txt"], PIP_LOG, echo=False)
    if rc != 0:
        C.say("\nERROR: installing the libraries failed. Last lines of the log:", LOG)
        for line in PIP_LOG.read_text(errors="replace").splitlines()[-12:]:
            print("  " + line)
        C.say(f"(full pip output: {PIP_LOG.relative_to(C.ROOT)})", LOG)
        C.say("\nCommon causes: no internet, or a VPN/proxy blocking pypi.org. Try another network.", LOG)
        C.say("Still failing? Run  python scripts/collect_diagnostics.py  and attach diagnostics.txt to a bug report.", LOG)
        return 1
    C.say("Libraries installed.", LOG)

    C.say("\nRunning the readiness checker ...\n", LOG)
    rc = C.run_logged([str(vp), "scripts/doctor.py"], LOG)
    if rc != 0:
        return 1
    C.say("\nSetup finished. Next step:   python scripts/quick_test.py", LOG)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
