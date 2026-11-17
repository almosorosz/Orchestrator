"""Shared wall-clock limits for interactive PBM analysis jobs."""

from threading import Event
from time import monotonic

from PySide6.QtWidgets import QDoubleSpinBox


def budget_field():
    field = QDoubleSpinBox()
    field.setDecimals(3)
    field.setRange(0., 1e9)
    field.setSpecialValueText("PE defaults")
    field.setSuffix(" s")
    field.setToolTip("Set a wall-clock limit for the entire run, including warm-up and compilation. "
                     "Zero uses the PE solver limits. JAX can stop once compilation returns control.")
    return field


class AnalysisBudget:
    def __init__(self):
        self.stop_requested = Event()
        self.deadline = float("inf")
        self.seconds = 0.
        # Keep the host callback identity stable for Diffrax's compilation cache.
        self.check = self.cancelled

    def start(self, seconds):
        self.seconds = seconds
        self.deadline = monotonic() + seconds if seconds > 0 else float("inf")
        self.stop_requested.clear()

    def stop(self):
        self.stop_requested.set()

    def expired(self):
        return not self.stop_requested.is_set() and monotonic() >= self.deadline

    def cancelled(self):
        return self.stop_requested.is_set() or self.expired()

    def apply(self, settings):
        if self.seconds:
            return {**settings, "MaxIterationTime": self.seconds}
        return settings
