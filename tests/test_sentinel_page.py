"""Integration checks for the PC client, using a local HTTP server and real Qt."""

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QUrl
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineCore import QWebEngineUrlRequestInterceptor

from gui.pages import sentinel_page


class SentinelHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.paths.append(self.path)
        if self.path == "/api/sensors":
            mode = self.server.mode
            if mode == "slow":
                time.sleep(0.6)
            if mode == "error":
                status, content_type, body = 503, "text/plain", b"Sentinel unavailable"
            elif mode == "invalid":
                status, content_type, body = 200, "application/json", b'{"wrong":true}'
            elif mode == "empty":
                status, content_type, body = 200, "application/json", b"[]"
            else:
                body = json.dumps([{"id": "probe-1", "temp_c": 24.5, "status": "ok"}] + getattr(self.server, "extra_sensors", [])).encode()
                status, content_type = 200, "application/json"
        elif self.path.startswith("/thermal_feed"):
            body = getattr(self.server, "thermal_jpeg", None)
            status, content_type = (200, "image/jpeg") if body else (503, "text/plain")
            body = body or b"Thermal unavailable"
        elif self.path in ("/api/userinfo", "/api/probeconfig"):
            status, content_type, body = 200, "application/json", b"{}"
        elif self.path == "/api/camera":
            status, content_type, body = 200, "application/json", b'{"available":false}'
        elif self.path.startswith("/api/eventlog?"):
            status, content_type, body = 200, "application/json", b"[]"
        elif self.path == "/api/eventlog.csv":
            status, content_type, body = 200, "text/csv", b"timestamp,sensor_id,level\n2026-10-17,probe-1,warn\n"
        else:
            status, content_type = 200, "text/html"
            body = self.server.dashboard or b'<html><body><h1>Sentinel test dashboard</h1><a href="/api/eventlog.csv" download>Download CSV</a></body></html>'
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if self.path == "/api/eventlog.csv":
            self.send_header("Content-Disposition", 'attachment; filename="fumehood_event_log.csv"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class LocalOnlyRequests(QWebEngineUrlRequestInterceptor):
    """The real dashboard test must never contact weather or font services."""

    def interceptRequest(self, info):
        if info.requestUrl().host() not in ("", "127.0.0.1"):
            info.block(True)


class SentinelPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # The Windows offscreen platform does not always discover system fonts.
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                cls.app.setFont(QFont(families[0], 10))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SentinelHandler)
        cls.server.daemon_threads = True
        cls.server.mode = "ok"
        cls.server.dashboard = None
        cls.server.paths = []
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.wifi_server = ThreadingHTTPServer(("127.0.0.1", 0), SentinelHandler)
        cls.wifi_server.daemon_threads = True
        cls.wifi_server.mode = "ok"
        cls.wifi_server.dashboard = None
        cls.wifi_server.paths = []
        cls.wifi_thread = threading.Thread(target=cls.wifi_server.serve_forever, daemon=True)
        cls.wifi_thread.start()
        cls.wifi_url = f"http://127.0.0.1:{cls.wifi_server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)
        cls.wifi_server.shutdown()
        cls.wifi_server.server_close()
        cls.wifi_thread.join(timeout=2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sentinel-client-test-")
        self.settings = QSettings(str(Path(self.temp.name) / "settings.ini"), QSettings.Format.IniFormat)
        self.page = sentinel_page.SentinelPage(settings=self.settings)
        self.page.POLL_INTERVAL_MS = 100
        self.page.REQUEST_TIMEOUT_MS = 150
        self.page._timer.setInterval(100)
        self.server.mode = "ok"
        self.server.dashboard = None
        self.server.paths.clear()
        self.server.extra_sensors = []
        self.server.thermal_jpeg = None
        self.wifi_server.mode = "ok"
        self.wifi_server.dashboard = None
        self.wifi_server.paths.clear()

    def tearDown(self):
        self.page.shutdown()
        self.page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()

    def wait_until(self, condition, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return
            QTest.qWait(10)
        self.fail("Timed out: " + self.page.status_label.text() + " / " + self.page.message_label.text())

    def connect(self):
        self.page.address_input.setText(self.url)
        self.page.connect_to_sentinel()

    def select_wifi(self):
        self.page.connection_combo.setCurrentIndex(self.page.connection_combo.findData("wifi"))

    def reopen_page(self):
        self.page.shutdown()
        self.page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.page = sentinel_page.SentinelPage(settings=self.settings)
        self.page.REQUEST_TIMEOUT_MS = 150
        self.page._timer.setInterval(100)

    def test_address_validation(self):
        self.assertEqual(sentinel_page.normalize_sentinel_url(" 192.168.77.2:8000/ "), sentinel_page.DEFAULT_SENTINEL_URL)
        self.assertEqual(sentinel_page.normalize_sentinel_url("https://sentinel.local:8443"), "https://sentinel.local:8443")
        for invalid in ("", "file:///tmp/dashboard.html", "ftp://sentinel", "http://sentinel/api/sensors", "http://user:pass@sentinel", "http://sentinel?q=1", "http://sentinel:0"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sentinel_page.normalize_sentinel_url(invalid)

    def test_legacy_ethernet_address_survives_upgrade(self):
        self.settings.setValue("sentinel/url", self.url)
        self.reopen_page()
        self.assertEqual(self.page.connection_combo.currentData(), "ethernet")
        self.assertEqual(self.page.address_input.text(), self.url)
        self.page.show()
        self.wait_until(lambda: self.page._page_loaded)
        self.assertEqual(self.settings.value("sentinel/ethernet_url"), self.url)

    def test_previous_name_settings_migrate_without_overwriting_new_values(self):
        old_values = {
            "url": self.wifi_url,
            "connection_mode": "wifi",
            "ethernet_url": self.url,
            "wifi_url": self.wifi_url,
            "download_directory": self.temp.name,
        }
        for key, value in old_values.items():
            self.settings.setValue(f"brain/{key}", value)
        self.settings.setValue("sentinel/download_directory", "new-download-folder")
        self.reopen_page()
        self.assertEqual(self.page.connection_combo.currentData(), "wifi")
        self.assertEqual(self.page.address_input.text(), self.wifi_url)
        self.assertEqual(self.settings.value("sentinel/ethernet_url"), self.url)
        self.assertEqual(self.settings.value("sentinel/download_directory"), "new-download-folder")
        for key in old_values:
            self.assertFalse(self.settings.contains(f"brain/{key}"))

    def test_wifi_requires_its_own_address(self):
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)
        self.select_wifi()
        self.assertEqual(self.page.address_input.text(), "")
        self.page.connect_to_sentinel()
        self.assertFalse(self.page._enabled)
        self.assertFalse(self.page._timer.isActive())
        self.assertIsNone(self.page._reply)
        self.assertIn("Enter the Pi", self.page.address_error_label.text())
        self.assertEqual(self.settings.value("sentinel/ethernet_url"), self.url)

    def test_connecting_wifi_first_after_upgrade_keeps_custom_ethernet(self):
        self.settings.setValue("sentinel/url", self.url)
        self.reopen_page()
        self.select_wifi()
        self.page.address_input.setText(self.wifi_url)
        self.page.connect_to_sentinel()
        self.wait_until(lambda: self.page._page_loaded)
        self.reopen_page()
        self.page.connection_combo.setCurrentIndex(0)
        self.assertEqual(self.page.address_input.text(), self.url)
        self.assertEqual(self.settings.value("sentinel/ethernet_url"), self.url)

    def test_profiles_save_separate_targets_and_restore_last_connection(self):
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)
        self.select_wifi()
        self.assertFalse(self.page._online)
        self.assertIs(self.page.content.currentWidget(), self.page.message_label)
        self.page.address_input.setText(self.wifi_url)
        self.page.connect_to_sentinel()
        self.wait_until(lambda: self.page._page_loaded)
        self.assertEqual(self.page.connection_combo.currentData(), "wifi")
        self.assertIn("/api/sensors", self.wifi_server.paths)
        self.assertIn("/", self.wifi_server.paths)
        self.assertEqual(self.settings.value("sentinel/ethernet_url"), self.url)
        self.assertEqual(self.settings.value("sentinel/wifi_url"), self.wifi_url)
        self.page.connection_combo.setCurrentIndex(0)
        self.assertEqual(self.page.address_input.text(), self.url)
        self.assertFalse(self.page._enabled)
        self.reopen_page()
        self.assertEqual(self.page.connection_combo.currentData(), "wifi")
        self.assertEqual(self.page.address_input.text(), self.wifi_url)
        self.page.show()
        self.wait_until(lambda: self.page._page_loaded)
        self.assertEqual(self.page._base_url, self.wifi_url)

    def test_wifi_first_does_not_overwrite_ethernet_default(self):
        self.select_wifi()
        self.page.address_input.setText(self.wifi_url)
        self.page.connect_to_sentinel()
        self.wait_until(lambda: self.page._page_loaded)
        self.reopen_page()
        self.page.connection_combo.setCurrentIndex(0)
        self.assertEqual(self.page.address_input.text(), sentinel_page.DEFAULT_SENTINEL_URL)

    def test_mode_change_cancels_old_request_before_connecting_wifi(self):
        self.server.mode = "slow"
        self.connect()
        self.select_wifi()
        self.page.address_input.setText(self.wifi_url)
        self.page.connect_to_sentinel()
        self.wait_until(lambda: self.page._page_loaded)
        # Keep yielding to the Python HTTP-server threads while late replies settle.
        deadline = time.monotonic() + 0.7
        self.wait_until(lambda: time.monotonic() >= deadline)
        self.assertTrue(self.page._online, self.page.message_label.text())
        self.assertEqual(self.page._base_url, self.wifi_url)
        self.assertEqual(self.page.connection_combo.currentData(), "wifi")

    def test_wifi_loss_and_recovery_preserve_ethernet_profile(self):
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)
        self.select_wifi()
        self.page.address_input.setText(self.wifi_url)
        self.page.connect_to_sentinel()
        self.wait_until(lambda: self.page._page_loaded)
        self.wifi_server.mode = "error"
        self.wait_until(lambda: not self.page._online)
        self.assertIn("Campus Wi-Fi", self.page.message_label.text())
        self.assertIn(self.wifi_url, self.page.message_label.text())
        self.wifi_server.mode = "ok"
        self.wait_until(lambda: self.page._page_loaded)
        self.assertEqual(self.settings.value("sentinel/ethernet_url"), self.url)
        self.assertEqual(self.page._base_url, self.wifi_url)

    def test_connect_loads_real_webview_and_saves_address(self):
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)
        self.assertTrue(self.page._online)
        self.assertIs(self.page.content.currentWidget(), self.page._browser)
        self.assertEqual(self.settings.value("sentinel/url"), self.url)
        self.assertIn("1 sensor", self.page.status_label.text())
        self.assertNotEqual(self.page.last_seen_label.text(), "Updated —")
        self.assertIn("/api/sensors", self.server.paths)
        self.assertIn("/", self.server.paths)

    def test_disconnect_hides_stale_page_then_automatically_recovers(self):
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)
        last_seen = self.page.last_seen_label.text()
        self.server.mode = "error"
        self.wait_until(lambda: not self.page._online)
        self.assertIs(self.page.content.currentWidget(), self.page.message_label)
        self.assertFalse(self.page._page_loaded)
        self.assertEqual(self.page.last_seen_label.text(), last_seen)
        self.server.mode = "ok"
        self.wait_until(lambda: self.page._page_loaded)
        self.assertTrue(self.page._online)

    def test_timeout_does_not_freeze_gui_and_can_recover(self):
        self.server.mode = "slow"
        self.connect()
        self.wait_until(lambda: self.page.status_label.text().startswith("Offline"))
        self.assertIs(self.page.content.currentWidget(), self.page.message_label)
        self.server.mode = "ok"
        self.wait_until(lambda: self.page._page_loaded)

    def test_invalid_api_is_not_treated_as_connected(self):
        self.server.mode = "invalid"
        self.connect()
        self.wait_until(lambda: "Invalid /api/sensors" in self.page.status_label.toolTip())
        self.assertFalse(self.page._online)
        self.assertIsNone(self.page._browser)

    def test_no_sensors_is_distinct_from_unreachable_server(self):
        self.server.mode = "empty"
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)
        self.assertTrue(self.page._online)
        self.assertIn("0 sensors", self.page.status_label.text())

    def test_manual_disconnect_stops_reconnection_and_discards_pending_reply(self):
        self.server.mode = "slow"
        self.connect()
        self.page.disconnect_from_sentinel()
        self.server.mode = "ok"
        QTest.qWait(700)
        self.assertFalse(self.page._online)
        self.assertFalse(self.page._timer.isActive())
        self.assertIsNone(self.page._reply)
        self.assertEqual("Disconnected", self.page.status_label.text())

    def test_new_address_cannot_receive_results_from_old_request(self):
        self.server.mode = "slow"
        self.connect()
        self.page.address_input.setText("http://127.0.0.1:1")
        self.page.connect_to_sentinel()
        self.wait_until(lambda: self.page.status_label.text().startswith("Offline"))
        QTest.qWait(700)
        self.assertFalse(self.page._online)
        self.assertIsNone(self.page._browser)

    def test_saved_address_is_reconnected_on_first_show(self):
        self.settings.setValue("sentinel/url", self.url)
        self.page.address_input.setText(self.url)
        self.page.show()
        self.wait_until(lambda: self.page._page_loaded)
        self.assertTrue(self.page._online)
        self.page.hide()

    def test_missing_webengine_preserves_external_browser_option(self):
        with patch.object(sentinel_page, "QWebEngineView", None):
            self.connect()
            self.wait_until(lambda: self.page._online)
            self.assertIn("Web view unavailable", self.page.message_label.text())
            with patch.object(sentinel_page.QDesktopServices, "openUrl", return_value=True) as opened:
                self.page.open_in_browser()
                opened.assert_called_once_with(QUrl(self.url + "/"))

    def test_actual_csv_download_and_cancel(self):
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)
        destination = Path(self.temp.name) / "events.csv"
        with patch.object(sentinel_page.QFileDialog, "getSaveFileName", return_value=(str(destination), "")):
            self.page._browser.page().download(QUrl(self.url + "/api/eventlog.csv"))
            self.wait_until(lambda: self.page.download_label.text().startswith("Saved:"))
        self.assertIn("probe-1,warn", destination.read_text())
        self.assertEqual(len(self.page._downloads), 0)
        destination.unlink()
        with patch.object(sentinel_page.QFileDialog, "getSaveFileName", return_value=("", "")) as dialog:
            self.page._browser.page().download(QUrl(self.url + "/api/eventlog.csv"))
            self.wait_until(lambda: dialog.called)
        QTest.qWait(100)
        self.assertFalse(destination.exists())

    def test_real_pi_dashboard_runs_inside_desktop_page(self):
        source = Path(__file__).resolve().parents[1] / "sentinel/pi/fumehood_dashboard.html"
        self.server.dashboard = source.read_bytes()
        if os.environ.get("SENTINEL_PREVIEW_CONNECTION") == "wifi":
            self.select_wifi()
        self.page._auto_started = True
        self.page.REQUEST_TIMEOUT_MS = 4000
        self.page._timer.setInterval(5000)
        self.page.resize(1100, 900)
        self.page.show()
        # Build the view with a local-only interceptor before its first navigation.
        self.page._base_url = self.url
        self.page._load_dashboard()
        self.interceptor = LocalOnlyRequests(self.page)
        self.page._profile.setUrlRequestInterceptor(self.interceptor)
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)
        readings = []
        def dashboard_ready():
            self.page._browser.page().runJavaScript(
                'JSON.stringify({lost:connectionLost,readings:sensors.length,tiles:document.getElementById("sensorTile").textContent})',
                lambda result: readings.append(result),
            )
            return readings and readings[-1] and json.loads(readings[-1])["readings"] == 1

        self.wait_until(dashboard_ready)
        data = json.loads(readings[-1])
        self.assertFalse(data["lost"])
        self.assertIn("24.50", data["tiles"])
        # Optionally save a rendered preview of the actual page for visual inspection.
        preview = os.environ.get("SENTINEL_PREVIEW_OUTPUT")
        if preview:
            QTest.qWait(1000)
            self.assertTrue(self.page._online)
            self.assertIs(self.page.content.currentWidget(), self.page._browser)
            self.assertTrue(self.page.grab().save(preview))

    def test_tc001_card_shows_heatmap_and_clears_offline_readings(self):
        try:
            import numpy as np
            from sentinel.pi.tc001_reader import decode_tc001_frame, render_thermal_jpeg
            import cv2
        except ImportError:
            self.skipTest("Install Pi thermal dependencies to run the TC001 browser integration check")
        words = np.full((192, 256), round((25 + 273.15) * 64), dtype=np.uint16)
        yy, xx = np.mgrid[:192, :256]
        warm = 25 + 50 * np.exp(-((xx - 130) ** 2 + (yy - 95) ** 2) / 1600)
        words[:] = np.rint((warm + 273.15) * 64).astype(np.uint16)
        frame = np.zeros((384, 256, 2), dtype=np.uint8)
        frame[192:, :, 0], frame[192:, :, 1] = words & 255, words >> 8
        temperatures, valid, stats = decode_tc001_frame(frame)
        self.server.thermal_jpeg = render_thermal_jpeg(temperatures, valid, stats)
        sensor = {"id": "thermal:tc001", "type": "thermal", "source": "topdon_tc001", "status": "ok", "temp_c": stats["max_c"], **stats}
        self.server.extra_sensors = [sensor]
        self.server.dashboard = (Path(__file__).resolve().parents[1] / "sentinel/pi/fumehood_dashboard.html").read_bytes()
        self.page._auto_started = True
        self.page.REQUEST_TIMEOUT_MS = 4000
        self.page._timer.setInterval(5000)
        self.page.resize(1100, 900)
        self.page.show()
        self.page._base_url = self.url
        self.page._load_dashboard()
        self.interceptor = LocalOnlyRequests(self.page)
        self.page._profile.setUrlRequestInterceptor(self.interceptor)
        self.connect()
        self.wait_until(lambda: self.page._page_loaded)

        def wait_js(expression):
            results = []

            def ready():
                self.page._browser.page().runJavaScript("Boolean(" + expression + ")", results.append)
                return results and results[-1] is True

            self.wait_until(ready)

        wait_js('sensors.some(s=>s.type==="thermal")')
        self.page._browser.page().runJavaScript('openView("sensors")')
        wait_js('document.querySelector(".thermal-card .thermal-name")?.textContent === "TOPDON TC001"')
        wait_js('document.getElementById("thermalThumb")?.naturalWidth > 0')
        preview = os.environ.get("SENTINEL_THERMAL_PREVIEW_OUTPUT")
        if preview:
            QTest.qWait(1200)
            self.assertTrue(self.page.grab().save(preview))
        self.page._browser.page().runJavaScript('openProbe("thermal:tc001")')
        wait_js('document.getElementById("thermalSummary")?.textContent.includes("Mean")')
        wait_js('document.getElementById("thermalBig")?.naturalWidth > 0')
        self.server.extra_sensors = [{**sensor, "status": "offline", "temp_c": None, "max_c": None}]
        self.page._browser.page().runJavaScript('pollSensors()')
        wait_js('document.getElementById("pCur")?.textContent === "—" && !document.getElementById("thermalBig")?.getAttribute("src")')
        self.page._browser.page().runJavaScript('goBack()')
        wait_js('document.querySelector(".thermal-card .thermal-value")?.textContent === "—" && document.querySelector(".thermal-state")?.textContent === "Offline"')


if __name__ == "__main__":
    unittest.main()
