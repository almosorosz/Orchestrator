"""Run fitting and recipe generation outside the dashboard thread."""
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class LoopModelJob(QThread):
    progress = Signal(str)
    stage = Signal(str, object)

    def __init__(self, session, manifest, parent):
        super().__init__(parent)
        self.session = session
        self.manifest = manifest
        self.collect_only = session.complete
        self.recommendation = None
        self.error = None

    def run(self):
        try:
            with (Path(self.manifest["model_path"]) / "model.log").open("a", encoding="utf-8") as log:
                def report(message):
                    log.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
                    log.flush()
                    self.progress.emit(message)

                try:
                    self.recommendation = self.session.recommend(
                        self.manifest, self.isInterruptionRequested, report,
                        publish=self.stage.emit, collect_only=self.collect_only)
                except Exception as error:
                    report(f"Modeling stopped: {error}")
                    raise
        except Exception as error:
            self.error = error
