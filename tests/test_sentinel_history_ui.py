"""Real History page, real HTTP/SQLite backup, and native PC controls; no hardware."""

import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QUrl
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from werkzeug.serving import make_server, WSGIRequestHandler

from gui.pages.sentinel_page import SentinelPage
from gui.sentinel_archive_ui import ArchiveSettingsDialog
from test_sentinel_page import LocalOnlyRequests
from test_sentinel_history import reading, HistoryStore, PI_DIR, utc_timestamp


class QuietHandler(WSGIRequestHandler):
    def log(self, *args, **kwargs):
        pass


class HistoryUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        font_id = QFontDatabase.addApplicationFont(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            cls.app.setFont(QFont(families[0], 10))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sentinel-history-ui-")
        self.root = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location("sentinel_history_ui_backend", PI_DIR / "app.py")
        self.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.backend)
        self.store = HistoryStore(self.root / "pi/history.sqlite3")
        self.backend.history_store = self.store
        self.backend._history_thread = MagicMock()
        self.backend.get_camera = lambda: None
        self.backend.read_all_sensors = lambda: [{"id": "probe-1", "type": "temp", "temp_c": 24.5, "status": "ok"}]
        self.store.add_system_event("Service started")
        self.store.capture([reading(), reading("thermal:tc001", 65, sensor_name="TOPDON TC001", type="thermal", min_c=20, max_c=65, mean_c=30, center_c=25, level="alert"),
                            reading("door-gp5", None, sensor_name="Sash", type="door", state="open", detail="Door open")])
        with self.store.connection() as con:
            for _ in range(210):
                self.store._insert(con, {**reading(), "kind": "readings", "timestamp": utc_timestamp()})
            self.store._insert(con, {**reading(), "kind": "readings", "timestamp": utc_timestamp(datetime.now(timezone.utc)-timedelta(days=3))})
        self.server = make_server("127.0.0.1", 0, self.backend.app, threaded=True, request_handler=QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        self.settings.setValue("sentinel/archive/directory", str(self.root / "pc"))
        (self.root / "cloud").mkdir()
        self.settings.setValue("sentinel/archive/cloud_enabled", True)
        self.settings.setValue("sentinel/archive/cloud_directory", str(self.root / "cloud"))
        self.page = SentinelPage(settings=self.settings)
        self.page._auto_started = True
        self.page.resize(1200, 900)
        self.page.show()
        self.page._base_url = self.url
        self.page._load_dashboard()
        self.interceptor = LocalOnlyRequests(self.page)
        self.page._profile.setUrlRequestInterceptor(self.interceptor)
        self.page.address_input.setText(self.url)
        self.page.connect_to_sentinel()
        self.wait(lambda: self.page._page_loaded)
        self.js('openView("mod3")')
        self.wait_js('logState.pi && logState.desktop && logState.desktop.pc === "Saved" && document.getElementById("logRows").children.length > 0')

    def tearDown(self):
        self.page.shutdown()
        self.page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def wait(self, condition, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return
            QTest.qWait(10)
        self.fail("Timed out: " + self.page.status_label.text())

    def js(self, expression):
        values = []
        self.page._browser.page().runJavaScript(expression, values.append)
        self.wait(lambda: bool(values))
        return values[0]

    def wait_js(self, expression):
        self.wait(lambda: bool(self.js("Boolean(" + expression + ")")))

    def test_history_filters_export_backup_and_native_settings(self):
        self.assertEqual(self.js('document.getElementById("logPiStatus").textContent'), "Recording · every 5s")
        self.assertIn("upload not verified", self.js('document.getElementById("logCloudStatus").textContent'))
        self.assertFalse(self.js('!document.getElementById("logSettings").hidden'))
        self.js('document.getElementById("logReadings").click()')
        self.wait_js('!logState.controller && logState.kind === "readings" && logState.rows.length === 200')
        self.js('document.getElementById("logMore").click()')
        self.wait_js('!logState.controller && logState.rows.length === 213')
        self.assertTrue(self.js("logState.older"))
        self.js('document.getElementById("logSensor").value="thermal:tc001";document.getElementById("logSensor").dispatchEvent(new Event("change"))')
        self.wait_js('!logState.controller && logState.rows.length === 1 && logState.rows[0].type === "thermal"')
        self.assertIn("Center 25", self.js('document.getElementById("logRows").textContent'))
        destination = self.root / "selected-thermal.csv"
        with patch("gui.pages.sentinel_page.QFileDialog.getSaveFileName", return_value=(str(destination), "")):
            self.js('document.getElementById("logExport").click()')
            self.wait(lambda: self.page.download_label.text().startswith("Saved:"))
        with destination.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["max_c"], "65.0")
        self.js('document.getElementById("logSettingsButton").click();document.getElementById("logInterval").value="10";document.getElementById("logSaveSettings").click()')
        self.wait(lambda: self.store.settings()["record_interval_seconds"] == 10)
        self.wait_js('document.getElementById("logSettingsNote").textContent === "Saved to Pi"')
        with patch.object(self.page, "show_log_settings") as configure:
            self.js('document.querySelector("a[href=\'/__sentinel_desktop__/settings\']").click()')
            self.wait(lambda: configure.called)
        # Capture actual rendering for review when requested, without recording hardware data.
        preview = os.environ.get("SENTINEL_HISTORY_PREVIEW")
        if preview:
            self.js('document.getElementById("logSettingsButton").click();document.getElementById("logEvents").click()')
            QTest.qWait(1200)
            self.page.grab().save(preview)
        dataset = self.store.status()["dataset_id"]
        self.assertTrue((self.root / "pc" / dataset / "archive.sqlite3").is_file())
        self.assertTrue((self.root / "cloud/Sentinel" / dataset / "history.sqlite3").is_file())

    def test_native_folder_settings_validate_and_custom_commands_need_pi_origin(self):
        dialog = ArchiveSettingsDialog(self.page._archive.config(), self.page)
        dialog.cloud_enabled.setChecked(True)
        dialog.cloud_directory.setText(str(self.root / "missing"))
        dialog.validate()
        self.assertIn("existing cloud sync folder", dialog.error.text())
        dialog.cloud_directory.setText(str(self.root / "cloud"))
        dialog.validate()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        self.assertEqual(dialog.values()["interval"], 60)
        self.assertFalse(self.page._is_pi_page(QUrl("https://example.com")))
        dialog.deleteLater()

    def test_pc_disconnect_and_restart_resume_without_duplicate_rows(self):
        from gui.sentinel_archive import ArchiveReplica
        dataset = self.store.status()["dataset_id"]
        replica = ArchiveReplica(self.root / "pc", dataset)
        original = replica.cursor()
        self.page.disconnect_from_sentinel()
        self.store.capture([reading(value=31, level="alert", detail="31.00 °C")])
        self.assertEqual(replica.cursor(), original)
        self.page.shutdown()
        self.page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.page = SentinelPage(settings=self.settings)
        self.page._auto_started = True
        self.page.address_input.setText(self.url)
        self.page.connect_to_sentinel()
        self.wait(lambda: self.page._archive.state()["pc"] == "Saved")
        self.assertEqual(replica.cursor(), self.store.status()["latest_seq"])
        with replica.connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM records").fetchone()[0], self.store.status()["record_count"])


if __name__ == "__main__":
    unittest.main()
