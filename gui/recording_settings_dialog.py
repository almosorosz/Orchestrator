import json
from pathlib import Path
from contextlib import closing
import sqlite3
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit, QCheckBox,
                               QPushButton, QDialogButtonBox, QMessageBox, QLabel, QFileDialog)
from app.paths import DATA_DIR
from experiment.recording import validate_settings

SETTINGS_FILE = DATA_DIR / "runtime" / "recording.json"


def save_recording_settings(settings):
    from experiment.recording import atomic_json
    settings = validate_settings(settings)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(SETTINGS_FILE, settings)


def load_recording_settings():
    if SETTINGS_FILE.exists():
        return validate_settings(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    return {"enabled": True, "sql_enabled": False, "output_folder": str(DATA_DIR / "runs"), "remote_sql_url": ""}


def database_counts(filename):
    path = Path(filename).resolve()
    if not path.exists():
        return None
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=.1)) as db:
        total = db.execute("SELECT count(*) FROM events").fetchone()[0]
        columns = {row[1] for row in db.execute("PRAGMA table_info(events)")}
        pending = db.execute("SELECT count(*) FROM events WHERE uploaded=0").fetchone()[0] if "uploaded" in columns else None
        return total, pending


class RecordingSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run recording and SQL")
        self.resize(820, 360)
        current = load_recording_settings()
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.sql_enabled = QCheckBox("Also save to SQL (local database and optional remote sync)")
        self.sql_enabled.setChecked(current.get("sql_enabled", True))
        self.folder = QLineEdit(current["output_folder"])
        self.sql = QLineEdit(current.get("remote_sql_url", ""))
        self.sql.setEchoMode(QLineEdit.EchoMode.Password)
        self.sql.setPlaceholderText("Optional: sqlite:///absolute/path/cloud.sqlite3 or postgresql+psycopg://…")
        form.addRow(QLabel("CSV recording is automatic for every experiment."))
        form.addRow(self.sql_enabled)
        form.addRow("Run output folder", self.folder)
        form.addRow("Remote SQL URL", self.sql)
        self.sql.setEnabled(self.sql_enabled.isChecked())
        self.sql_enabled.toggled.connect(self.sql.setEnabled)
        layout.addLayout(form)
        demo = QPushButton("Choose local demo SQL destination…")
        demo.clicked.connect(self.choose_demo)
        layout.addWidget(demo)
        status_button = QPushButton("Refresh SQL status")
        status_button.clicked.connect(self.refresh_status)
        layout.addWidget(status_button)
        self.status = QLabel("Save changes, run an experiment, then refresh SQL status here.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choose_demo(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder for cloud_demo.sqlite3")
        if folder:
            self.sql_enabled.setChecked(True)
            self.sql.setText("sqlite:///" + (Path(folder).resolve() / "cloud_demo.sqlite3").as_posix())

    def refresh_status(self):
        try:
            counts = database_counts(Path(self.folder.text().strip()) / "recordings.sqlite3")
            text = "Local SQL database: not created yet." if counts is None else f"Local SQL records: {counts[0]}."
            url = self.sql.text().strip()
            if url and self.sql_enabled.isChecked() and counts is not None:
                text += f" Awaiting upload: {counts[1]}."
            elif not url:
                text += " Remote synchronization is off (local only)."
            if url.startswith("sqlite:///"):
                remote = database_counts(url[len("sqlite:///"):])
                text += " Demo destination: not created yet." if remote is None else f" Demo SQL records: {remote[0]}."
            elif url:
                text += " Server synchronization progress is shown by the awaiting-upload count."
            if not self.sql_enabled.isChecked():
                text += " SQL is switched off; existing databases are retained."
            self.status.setText(text)
        except Exception as error:
            self.status.setText(f"SQL status unavailable: {type(error).__name__}")

    def save(self):
        settings = {
            'enabled': True, 'output_folder': self.folder.text().strip(),
            'sql_enabled': self.sql_enabled.isChecked(), 'remote_sql_url': self.sql.text().strip(),
        }
        try:
            save_recording_settings(settings)
            self.accept()
        except Exception as error:
            QMessageBox.warning(self, "Invalid recording settings", str(error))
