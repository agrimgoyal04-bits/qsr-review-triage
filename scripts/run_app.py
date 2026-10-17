"""Start the API (background) and the web UI (foreground). Ctrl+C stops both.

    python scripts/run_app.py
    COMPLAINT_THRESHOLD=0.8 python scripts/run_app.py            (macOS / Linux)
    set COMPLAINT_THRESHOLD=0.8 && python scripts\\run_app.py     (Windows cmd)
"""
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.ensure_venv()
API_PORT = 8000


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def api_ready() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}/health", timeout=2) as r:
            return b'"ok"' in r.read()
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    for m in ("gatekeeper", "router"):
        if not (C.ROOT / "models" / f"{m}_onnx" / "model.onnx").exists():
            print(f"ERROR: models/{m}_onnx/model.onnx not found. Train first:   python scripts/train_all.py medium")
            return 1
    if port_in_use(API_PORT):
        print(f"ERROR: something is already using port {API_PORT} (an earlier run may still be open).")
        print("Close the other terminal running this, then try again.")
        return 1

    C.LOGS.mkdir(exist_ok=True)
    api_log = open(C.LOGS / "api.log", "wb")
    print(f"Starting API on http://127.0.0.1:{API_PORT} (log: logs/api.log) ...")
    api = subprocess.Popen(C.python_cmd("-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)),
                           cwd=str(C.ROOT), stdout=api_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if api_ready():
                break
            if api.poll() is not None:
                print("The API crashed on start. Last log lines:")
                print((C.LOGS / "api.log").read_text(errors="replace")[-1500:])
                return 1
            time.sleep(1)
        else:
            print("The API did not become ready in 120 s. See logs/api.log")
            return 1

        # Streamlit asks for an email on its very first start, which looks like a frozen terminal. Skip it.
        cred = Path.home() / ".streamlit" / "credentials.toml"
        if not cred.exists():
            cred.parent.mkdir(exist_ok=True)
            cred.write_text('[general]\nemail = ""\n')

        print("API is ready. Opening the web page (if it does not open, visit the URL printed below) ...")
        subprocess.call(C.python_cmd("-m", "streamlit", "run", "app.py", "--browser.gatherUsageStats", "false"), cwd=str(C.ROOT))
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        print("\nStopping API ...")
        api.terminate()
        try:
            api.wait(10)
        except subprocess.TimeoutExpired:
            api.kill()
        api_log.close()


if __name__ == "__main__":
    sys.exit(main())
