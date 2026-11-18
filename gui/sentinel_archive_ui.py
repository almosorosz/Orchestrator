"""Background PC backup controller and compact native settings dialog."""

from pathlib import Path
import time

from PySide6.QtCore import QObject, QStandardPaths, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
)

from gui.sentinel_archive import replicate


class ArchiveWorker(QThread):
    updated = Signal(dict)
    completed = Signal(dict)

    def __init__(self, url, directory, cloud_directory, cloud_due, parent=None):
        super().__init__(parent)
        self.arguments = (url, directory, cloud_directory)
        self.cloud_due = cloud_due

    def run(self):
        try:
            result = replicate(*self.arguments, cloud_due=self.cloud_due,
                               cancelled=self.isInterruptionRequested, progress=self.updated.emit)
        except Exception as error:
            result = {"pc": "Backup failed · retrying", "error": str(error)}
        self.completed.emit(result)


class ArchiveController(QObject):
    changed = Signal(dict)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._url = ""
        self._online = False
        self._worker = None
        self._epoch = 0
        self._closed = False
        self._cloud_time = 0
        self._force_cloud_pending = False
        self._state = {"pc": "Waiting for Pi", "cloud": "Off", "error": ""}
        self._timer = QTimer(self)
        self._timer.setInterval(self.config()["interval"] * 1000)
        self._timer.timeout.connect(self.sync_now)
        self._timer.start()

    def config(self):
        def boolean(key, default):
            return str(self.settings.value("sentinel/archive/" + key, default)).lower() in ("true", "1")
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        return {
            "enabled": boolean("enabled", True),
            "directory": str(self.settings.value("sentinel/archive/directory", str(Path(documents) / "Sentinel Logs"))),
            "cloud_enabled": boolean("cloud_enabled", False),
            "cloud_directory": str(self.settings.value("sentinel/archive/cloud_directory", "")),
            "interval": max(10, min(3600, int(self.settings.value("sentinel/archive/interval", 60)))),
        }

    def state(self):
        config = self.config()
        return {**self._state, "available": True,
                "pc": self._state["pc"] if config["enabled"] else "Off",
                "cloud": self._state.get("cloud", "Waiting") if config["cloud_enabled"] else "Off",
                "pc_directory": config["directory"], "cloud_target": config["cloud_directory"],
                "cloud_note": "A cloud sync app must upload this folder; remote upload is not verified."}

    def _emit(self):
        self.changed.emit(self.state())

    def set_source(self, url, online):
        changed = url != self._url
        recovered = online and not self._online
        if not online and self._online:
            self._epoch += 1
            if self._worker:
                self._worker.requestInterruption()
        if changed:
            self._epoch += 1
            self._cloud_time = 0
            self._state = {"pc": "Waiting for Pi", "cloud": "Waiting", "error": ""}
            if self._worker:
                self._worker.requestInterruption()
        self._url, self._online = url, online
        if not online:
            self._state["pc"] = "Waiting for Pi"
            self._emit()
        elif changed or recovered:
            self.sync_now()

    def sync_now(self, *, force_cloud=False):
        config = self.config()
        if self._closed or not (config["enabled"] or config["cloud_enabled"]):
            return
        if not self._online or not self._url:
            self._state["pc"] = "Waiting for Pi"
            self._emit()
            return
        if self._worker is not None:
            self._force_cloud_pending = self._force_cloud_pending or force_cloud
            return
        directory = config["directory"]
        if not config["enabled"]:
            directory = str(Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)) / "Sentinel spool")
        cloud = config["cloud_directory"] if config["cloud_enabled"] else ""
        if config["cloud_enabled"] and not cloud:
            self._state.update(cloud="Choose a cloud folder", error="Choose a folder in Log settings")
            self._emit()
            return
        self._state.update(pc="Saving…", error="", more=False)
        self._emit()
        worker = ArchiveWorker(self._url, directory, cloud, force_cloud or self._force_cloud_pending or not self._cloud_time or time.monotonic() - self._cloud_time >= 300, self)
        self._force_cloud_pending = False
        self._worker = worker
        epoch = self._epoch
        worker.updated.connect(lambda result: self._updated(result, epoch), Qt.ConnectionType.QueuedConnection)
        worker.completed.connect(lambda result: self._completed(result, epoch), Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(lambda: self._finished(worker, epoch), Qt.ConnectionType.QueuedConnection)
        worker.start()

    def _updated(self, result, epoch):
        if epoch == self._epoch and not self._closed:
            # Progress must not reset a successful previous cloud-folder save.
            result = {key: value for key, value in result.items() if key != "cloud"}
            self._state.update(result)
            self._emit()

    def _completed(self, result, epoch):
        if epoch != self._epoch or self._closed:
            return
        if result.get("cloud_saved"):
            self._cloud_time = time.monotonic()
            result["cloud_saved_at"] = time.strftime("%H:%M:%S")
        elif result.get("cloud") == "Waiting" and self._state.get("cloud_saved_at"):
            result.pop("cloud")
        if result.get("pc") == "Saved":
            result["saved_at"] = time.strftime("%H:%M:%S")
        if result.get("error"):
            result["more"] = False
        self._state.update(result)
        self._emit()

    def _finished(self, worker, epoch):
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()
        if not self._closed and self._online and (epoch != self._epoch or self._state.get("more") or self._force_cloud_pending):
            QTimer.singleShot(500, self.sync_now)

    def configure(self, parent):
        dialog = ArchiveSettingsDialog(self.config(), parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for key, value in dialog.values().items():
            self.settings.setValue("sentinel/archive/" + key, value)
        self.settings.sync()
        self._epoch += 1
        self._cloud_time = 0
        self._state = {"pc": "Waiting for Pi", "cloud": "Waiting", "error": ""}
        if self._worker:
            self._worker.requestInterruption()
        self._timer.setInterval(self.config()["interval"] * 1000)
        self._emit()
        self.sync_now(force_cloud=True)

    def shutdown(self):
        self._closed = True
        self._timer.stop()
        if self._worker:
            self._worker.requestInterruption()
            # Never destroy a running QThread. HTTP reads have a bounded timeout.
            self._worker.wait()


class ArchiveSettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sentinel — Log settings")
        self.resize(570, 280)
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox("Automatically save logs to this PC")
        self.enabled.setChecked(config["enabled"])
        layout.addWidget(self.enabled)
        form = QFormLayout()
        self.directory = self.folder_row(form, "PC folder", config["directory"])
        self.interval = QSpinBox()
        self.interval.setRange(10, 3600)
        self.interval.setSuffix(" seconds")
        self.interval.setValue(config["interval"])
        form.addRow("Save interval", self.interval)
        layout.addLayout(form)
        self.cloud_enabled = QCheckBox("Also save a copy to a cloud sync folder")
        self.cloud_enabled.setChecked(config["cloud_enabled"])
        layout.addWidget(self.cloud_enabled)
        cloud_form = QFormLayout()
        self.cloud_directory = self.folder_row(cloud_form, "Cloud folder", config["cloud_directory"])
        layout.addLayout(cloud_form)
        note = QLabel("Select a folder managed by your cloud sync app. Copies refresh during a save, at most once every 5 minutes. "
                      "The app handles uploading; Sentinel cannot verify remote uploads. "
                      "Orchestrator must stay open and connected to the Pi.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error = QLabel("")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def folder_row(self, form, label, value):
        row = QHBoxLayout()
        field = QLineEdit(value)
        button = QPushButton("Browse…")
        def browse():
            chosen = QFileDialog.getExistingDirectory(self, label, field.text())
            if chosen:
                field.setText(chosen)
        button.clicked.connect(browse)
        row.addWidget(field, 1)
        row.addWidget(button)
        form.addRow(label, row)
        return field

    def validate(self):
        for enabled, field in ((self.enabled.isChecked(), self.directory), (self.cloud_enabled.isChecked(), self.cloud_directory)):
            if enabled and (not field.text().strip() or not Path(field.text().strip()).is_absolute()):
                self.error.setText("Choose an absolute folder path for each enabled destination.")
                return
        if self.cloud_enabled.isChecked() and not Path(self.cloud_directory.text().strip()).is_dir():
            self.error.setText("Choose an existing cloud sync folder.")
            return
        self.accept()

    def values(self):
        return {"enabled": self.enabled.isChecked(), "directory": self.directory.text().strip(),
                "cloud_enabled": self.cloud_enabled.isChecked(), "cloud_directory": self.cloud_directory.text().strip(),
                "interval": self.interval.value()}
