"""Detects the "training gets slower and slower" problem (seen on Apple GPUs / MPS).

Feed it (step, time) every step; it measures seconds-per-step in windows, learns a baseline from the
first few healthy windows, and returns a warning message once if speed stays much worse than baseline.
Pure Python (no torch) so it is easy to test.
"""
import statistics
from typing import Optional


class SlowdownDetector:
    def __init__(self, interval: int = 25, factor: float = 4.0, patience: int = 3,
                 skip_windows: int = 1, baseline_windows: int = 4):
        self.interval, self.factor, self.patience = interval, factor, patience
        self.skip_windows, self.baseline_windows = skip_windows, baseline_windows
        self._last_step: Optional[int] = None
        self._last_time: Optional[float] = None
        self._seen = 0                      # windows measured so far
        self._baseline_rates: list = []
        self.baseline: Optional[float] = None
        self._bad = 0
        self.warned = False
        self.last_rate: Optional[float] = None

    def update(self, step: int, now: float) -> Optional[str]:
        if self._last_step is None:
            self._last_step, self._last_time = step, now
            return None
        if step - self._last_step < self.interval:
            return None
        rate = (now - self._last_time) / (step - self._last_step)  # seconds per step in this window
        self._last_step, self._last_time = step, now
        self.last_rate = rate
        self._seen += 1
        if self._seen <= self.skip_windows:      # first window includes one-off startup cost
            return None
        if self.baseline is None:
            self._baseline_rates.append(rate)
            if len(self._baseline_rates) >= self.baseline_windows:
                self.baseline = statistics.median(self._baseline_rates)
            return None
        self._bad = self._bad + 1 if rate > self.factor * self.baseline else 0
        if self._bad >= self.patience and not self.warned:
            self.warned = True
            return (
                f"\n*** SLOWDOWN DETECTED at step {step}: now {rate:.1f} s/step vs {self.baseline:.1f} s/step "
                f"at the start ({rate / self.baseline:.0f}x slower). ***\n"
                "Training is not broken, but it will take far too long. Press Ctrl+C and see README section 7, "
                "'Training starts fast then slows down' (docs/GUIDE.md). Include the lines starting with [mem] if you report this."
            )
        return None
