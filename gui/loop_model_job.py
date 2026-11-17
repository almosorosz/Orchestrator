"""Run fitting and recipe generation outside the dashboard thread."""
from PySide6.QtCore import QThread, Signal


class LoopModelJob(QThread):
    progress = Signal(str)

    def __init__(self, session, manifest, parent):
        super().__init__(parent)
        self.session = session
        self.manifest = manifest
        self.recommendation = None
        self.error = None

    def run(self):
        try:
            self.recommendation = self.session.recommend(
                self.manifest, self.isInterruptionRequested, self.progress.emit)
        except Exception as error:
            self.error = error
