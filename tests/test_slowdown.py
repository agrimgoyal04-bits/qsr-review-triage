import pytest

from src.slowdown import SlowdownDetector


def feed(det, secs_per_step, steps, start_step=0, start_time=0.0):
    """Feed constant-speed steps; return (list of messages, final step, final time)."""
    msgs, t, s = [], start_time, start_step
    for _ in range(steps):
        s += 1
        t += secs_per_step
        m = det.update(s, t)
        if m:
            msgs.append(m)
    return msgs, s, t


def test_no_warning_when_speed_is_steady():
    d = SlowdownDetector()
    msgs, *_ = feed(d, 1.8, 1000)
    assert msgs == [] and d.baseline == pytest.approx(1.8, rel=0.05)


def test_warns_once_on_sustained_slowdown():
    d = SlowdownDetector()
    m1, s, t = feed(d, 1.8, 300)          # healthy start (learns baseline)
    m2, s, t = feed(d, 35.0, 200, s, t)   # the real-world case: 1.8 -> 35 s/it
    assert m1 == [] and len(m2) == 1
    assert "SLOWDOWN DETECTED" in m2[0] and "35.0 s/step" in m2[0]
    m3, *_ = feed(d, 35.0, 200, s, t)     # keeps being slow: no repeated spam
    assert m3 == []


def test_single_stall_does_not_warn():
    d = SlowdownDetector()
    _, s, t = feed(d, 1.8, 300)
    t += 600.0                            # e.g. the computer slept for 10 minutes: one slow window only
    msgs, s, t = feed(d, 1.8, 300, s, t)
    assert msgs == []


def test_modest_variation_does_not_warn():
    d = SlowdownDetector()
    _, s, t = feed(d, 1.8, 300)
    msgs, *_ = feed(d, 5.0, 300, s, t)    # 2.8x slower: below the 4x threshold
    assert msgs == []


def test_startup_cost_in_first_window_is_ignored():
    d = SlowdownDetector()
    t = 0.0
    t += 120.0                             # long first-step compile/startup
    d.update(1, t)
    msgs, *_ = feed(d, 1.8, 400, 1, t)
    assert msgs == [] and d.baseline == pytest.approx(1.8, rel=0.1)
