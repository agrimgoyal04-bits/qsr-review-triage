"""Tests for the cross-platform launcher helpers (scripts/_common.py, scripts/setup_env.py).

The OS-specific branches are pure functions, so Windows/macOS behaviour is tested here on any OS.
"""
import sys
from pathlib import Path

import pytest

import _common as C
import setup_env


# ---------------------------------------------------------------- venv path per OS
def test_venv_python_paths():
    root = Path("/proj")
    assert C.venv_python(root, windows=False) == Path("/proj/.venv/bin/python")
    assert C.venv_python(root, windows=True) == Path("/proj/.venv/Scripts/python.exe")


@pytest.mark.parametrize("ver,ok", [((3, 12), True), ((3, 13), True), ((3, 11), False), ((3, 14), False), ((3, 12, 5), True)])
def test_supported_python(ver, ok):
    assert C.is_supported_python(ver) is ok


# ---------------------------------------------------------------- keep-awake wrapper
def test_keep_awake_macos():
    assert C.keep_awake_prefix("darwin", which=lambda n: "/usr/bin/caffeinate") == ["caffeinate", "-i"]
    assert C.keep_awake_prefix("darwin", which=lambda n: None) == []


def test_keep_awake_linux_uses_inhibit_only_if_it_works():
    have = lambda n: "/usr/bin/systemd-inhibit"  # noqa: E731
    pre = C.keep_awake_prefix("linux", which=have, probe=lambda p: True)
    assert pre[0] == "systemd-inhibit" and "--what=idle:sleep" in pre
    assert C.keep_awake_prefix("linux", which=have, probe=lambda p: False) == []  # e.g. inside WSL/containers
    assert C.keep_awake_prefix("linux", which=lambda n: None) == []


def test_keep_awake_windows_has_no_wrapper():
    assert C.keep_awake_prefix("win32", which=lambda n: "x") == []


# ---------------------------------------------------------------- presets
def test_presets_have_both_stages_and_known_flags():
    for name, p in C.PRESETS.items():
        assert set(p) == {"gatekeeper", "router"}, name


def test_extra_flags_come_last_so_they_override():
    args = C.preset_args("medium", "gatekeeper", ["--epochs", "3", "--cpu"])
    assert args[-3:] == ["--epochs", "3", "--cpu"]
    assert args.index("--epochs") < len(args) - 3  # preset's own --epochs appears earlier (argparse: last one wins)


def test_unknown_preset():
    with pytest.raises(KeyError, match="Unknown preset"):
        C.preset_args("huge", "router")


def test_sample_pass_constants_match_dataset_sizes():
    assert C.SAMPLE_PASSES["medium"] == 30000 + 2 * 20000
    assert C.SAMPLE_PASSES["full"] == 104854 * 2 + 63757 * 3


# ---------------------------------------------------------------- log parsing / estimates
LOG = """>>> Training device: MPS
{'train_runtime': '18.39', 'train_samples_per_second': '17.41', 'epoch': '1'}
{'train_runtime': '9.2', 'train_samples_per_second': '17.33', 'epoch': '1'}
"""


def test_parse_training_log():
    dev, speeds = C.parse_training_log(LOG)
    assert dev == "MPS" and speeds == [17.41, 17.33]
    assert C.parse_training_log("nothing here") == (None, [])


def test_estimate_hours():
    assert C.estimate_hours(36000, 10.0) == pytest.approx(1.0)


# ---------------------------------------------------------------- python discovery
def test_candidates_include_windows_launcher_only_on_windows():
    nt = setup_env.candidate_commands("nt")
    posix = setup_env.candidate_commands("posix")
    assert ["py", "-3.12"] in nt and ["py", "-3.12"] not in posix
    assert nt[0] == [sys.executable] and ["python3.12"] in posix


def test_probe_current_interpreter():
    ver = setup_env.probe([sys.executable])
    assert ver == sys.version_info[:2]
    assert setup_env.probe(["definitely-not-a-real-python"]) is None


# ---------------------------------------------------------------- run_logged
def test_run_logged_captures_output_and_exit_code(tmp_path):
    log = tmp_path / "x.log"
    rc = C.run_logged([sys.executable, "-c", "import sys; print('hello'); sys.exit(3)"], log, echo=False)
    assert rc == 3 and "hello" in log.read_text()


def test_run_logged_preserves_carriage_returns_for_progress_bars(tmp_path):
    log = tmp_path / "y.log"
    code = "import sys; sys.stdout.write('10%\\r50%\\r100%\\n'); sys.stdout.flush()"
    C.run_logged([sys.executable, "-c", code], log, echo=False)
    assert log.read_bytes().count(b"\r") == 2


# ---------------------------------------------------------------- ensure_venv
def test_ensure_venv_exits_with_help_when_no_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "venv_python", lambda *a, **k: tmp_path / "nope" / "python")
    with pytest.raises(SystemExit) as e:
        C.ensure_venv()
    assert "setup_env.py" in str(e.value)


def test_ensure_venv_is_noop_inside_the_venv(monkeypatch, tmp_path):
    fake = tmp_path / "python"
    fake.write_text("")
    monkeypatch.setattr(C, "venv_python", lambda *a, **k: fake)
    monkeypatch.setattr(C, "running_in_project_venv", lambda *a, **k: True)
    assert C.ensure_venv() is None


def test_ensure_venv_hands_over_with_exec_on_posix(monkeypatch, tmp_path):
    fake = tmp_path / "python"
    fake.write_text("")
    calls = []
    monkeypatch.setattr(C, "venv_python", lambda *a, **k: fake)
    monkeypatch.setattr(C, "running_in_project_venv", lambda *a, **k: False)
    monkeypatch.setattr(C.os, "name", "posix")
    monkeypatch.setattr(C.os, "execv", lambda path, argv: calls.append((path, argv)))
    C.ensure_venv()
    assert calls and calls[0][0] == str(fake) and calls[0][1][0] == str(fake)
