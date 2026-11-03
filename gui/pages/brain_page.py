"""Desktop client for the Sentinel dashboard served by a Raspberry Pi."""

import json
from pathlib import Path

from PySide6.QtCore import QDateTime, QSettings, QStandardPaths, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)
from gui.sentinel_archive_ui import ArchiveController

try:
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None


DEFAULT_BRAIN_URL = "http://192.168.77.2:8000"


if QWebEngineView is not None:
    class SentinelWebPage(QWebEnginePage):
        """Handle explicit History actions only from the connected Pi origin."""
        def __init__(self, profile, browser, owner):
            super().__init__(profile, browser)
            self.owner = owner

        def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
            if self.owner._is_pi_page(url) and url.path().startswith("/__sentinel_desktop__/"):
                if (is_main_frame and navigation_type == self.NavigationType.NavigationTypeLinkClicked
                        and self.owner._is_pi_page(self.url())):
                    if url.path() == "/__sentinel_desktop__/settings":
                        QTimer.singleShot(0, self.owner.show_log_settings)
                    elif url.path() == "/__sentinel_desktop__/sync":
                        QTimer.singleShot(0, lambda: self.owner._archive.sync_now(force_cloud=True))
                return False
            return super().acceptNavigationRequest(url, navigation_type, is_main_frame)


def normalize_brain_url(address):
    """Accept a server origin, never a local file or an API/page path."""
    address = address.strip()
    if not address:
        raise ValueError("Enter the Pi address, including :8000.")
    if "://" not in address:
        address = "http://" + address
    url = QUrl(address, QUrl.ParsingMode.StrictMode)
    if (
        not url.isValid() or url.scheme() not in ("http", "https")
        or not url.host() or url.userInfo() or url.hasQuery() or url.hasFragment()
        or url.path() not in ("", "/") or url.port() == 0
    ):
        raise ValueError("Use http://host:port with no page path.")
    url.setPath("")
    return url.toString().rstrip("/")


class BrainPage(QWidget):
    POLL_INTERVAL_MS = 5000
    REQUEST_TIMEOUT_MS = 4000

    def __init__(self, parent=None, *, settings=None):
        super().__init__(parent)
        self._settings = settings if settings is not None else QSettings("Crystalloop", "Orchestrator")
        saved_mode = str(self._settings.value("brain/connection_mode", "ethernet"))
        self._connection_mode = saved_mode if saved_mode in ("ethernet", "wifi") else "ethernet"
        # The original single-address setting belongs to the Ethernet profile.
        # Once profiles exist, brain/url is only the last used address, not a fallback.
        legacy_url = (
            self._settings.value("brain/url", DEFAULT_BRAIN_URL)
            if not self._settings.contains("brain/connection_mode") else DEFAULT_BRAIN_URL
        )
        self._addresses = {
            "ethernet": str(self._settings.value("brain/ethernet_url", legacy_url)),
            "wifi": str(self._settings.value("brain/wifi_url", "")),
        }
        if (
            self._settings.contains("brain/url")
            and not self._settings.contains("brain/connection_mode")
            and not self._settings.contains("brain/ethernet_url")
        ):
            # Preserve a custom legacy address even if Wi-Fi is used first after upgrade.
            self._settings.setValue("brain/ethernet_url", self._addresses["ethernet"])
        self._base_url = ""
        self._enabled = False
        self._online = False
        self._generation = 0
        self._reply = None
        self._page_loaded = False
        self._page_loading = False
        self._browser = None
        self._downloads = set()
        self._auto_started = False

        self._network = QNetworkAccessManager(self)
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._check_connection)
        self._build_ui()
        self._archive = ArchiveController(self._settings, self)
        self._archive.changed.connect(self._push_archive_state)
        self._archive_timer = QTimer(self)
        self._archive_timer.setInterval(2000)
        self._archive_timer.timeout.connect(self._push_archive_state)
        self._archive_timer.start()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("Sentinel")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        header.addWidget(title)
        header.addStretch()
        self.status_label = QLabel("Disconnected")
        header.addWidget(self.status_label)
        self.setup_button = QPushButton("Help")
        self.setup_button.clicked.connect(self.show_connection_help)
        header.addWidget(self.setup_button)
        layout.addLayout(header)

        connection_row = QHBoxLayout()
        self.connection_combo = QComboBox()
        self.connection_combo.addItem("Ethernet", "ethernet")
        self.connection_combo.addItem("Wi-Fi", "wifi")
        self.connection_combo.setToolTip("Connection type")
        self.connection_combo.setCurrentIndex(self.connection_combo.findData(self._connection_mode))
        connection_row.addWidget(self.connection_combo)
        self.address_input = QLineEdit()
        self.address_input.setAccessibleName("Pi address")
        self.address_input.setText(self._addresses[self._connection_mode])
        self.address_input.returnPressed.connect(self.connect_to_brain)
        connection_row.addWidget(self.address_input, 1)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_to_brain)
        connection_row.addWidget(self.connect_button)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)
        self.disconnect_button.clicked.connect(self.disconnect_from_brain)
        connection_row.addWidget(self.disconnect_button)
        layout.addLayout(connection_row)
        self.address_error_label = QLabel("")
        self.address_error_label.setWordWrap(True)
        self.address_error_label.setStyleSheet("color: #d97706;")
        self.address_error_label.hide()
        layout.addWidget(self.address_error_label)

        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setEnabled(False)
        self.refresh_button.clicked.connect(self.refresh_dashboard)
        actions.addWidget(self.refresh_button)
        self.external_button = QPushButton("Open in browser")
        self.external_button.clicked.connect(self.open_in_browser)
        actions.addWidget(self.external_button)
        self.logs_button = QPushButton("Log settings")
        self.logs_button.clicked.connect(self.show_log_settings)
        actions.addWidget(self.logs_button)
        actions.addStretch()
        self.last_seen_label = QLabel("Updated —")
        self.last_seen_label.setToolTip("Last successful sensor check")
        actions.addWidget(self.last_seen_label)
        layout.addLayout(actions)
        self.content = QStackedWidget()
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content.addWidget(self.message_label)
        layout.addWidget(self.content, 1)
        self.download_label = QLabel("")
        self.download_label.setWordWrap(True)
        layout.addWidget(self.download_label)
        self._apply_connection_mode()
        self.connection_combo.currentIndexChanged.connect(self._change_connection_mode)

    @property
    def _connection_name(self):
        return "Wi-Fi" if self._connection_mode == "wifi" else "Ethernet"

    def _apply_connection_mode(self):
        if self._connection_mode == "wifi":
            self.address_input.setPlaceholderText("http://<Pi Wi-Fi IP>:8000")
        else:
            self.address_input.setPlaceholderText(DEFAULT_BRAIN_URL)
        self._show_message("Connect to view Sentinel.")

    def _change_connection_mode(self, index):
        mode = self.connection_combo.itemData(index)
        if mode not in self._addresses or mode == self._connection_mode:
            return
        self._addresses[self._connection_mode] = self.address_input.text()
        self.disconnect_from_brain()
        self._base_url = ""
        self._auto_started = True
        self._connection_mode = mode
        self.address_input.setText(self._addresses[mode])
        self.address_input.setToolTip("")
        self.address_error_label.hide()
        self.last_seen_label.setText("Updated —")
        self.last_seen_label.setToolTip("Last successful sensor check")
        self._apply_connection_mode()
        self._set_status("Disconnected", "#d97706")

    def show_connection_help(self):
        if self._connection_mode == "wifi":
            instructions = (
                "1. Join Wi-Fi on the Pi (Automatic / DHCP).\n"
                "2. Find its IP: ip -4 addr show wlan0\n"
                "3. Run python app.py on the Pi.\n"
                "4. Enter http://PI_IP:8000 and connect.\n\n"
                "The PC must be able to reach the Pi. Campus Wi-Fi may block device access.\n"
                "Check the IP again if it changes. Ethernet settings are saved separately."
            )
        else:
            instructions = (
                "1. Connect the Pi and PC by cable; power the Pi separately.\n"
                "2. PC: 192.168.77.1 / Pi: 192.168.77.2\n"
                "3. Mask: 255.255.255.0; gateway and DNS blank.\n"
                "4. Run python app.py on the Pi.\n"
                "5. Connect to http://192.168.77.2:8000.\n\n"
                "Use another subnet if this one is already in use. Wi-Fi settings are saved separately."
            )
        QMessageBox.information(self, f"Sentinel — {self._connection_name}", instructions)

    def showEvent(self, event):
        super().showEvent(event)
        # Only reconnect automatically to an address the user has saved by connecting.
        if not self._auto_started:
            self._auto_started = True
            has_saved_address = self._settings.contains(f"brain/{self._connection_mode}_url")
            has_legacy_address = self._connection_mode == "ethernet" and self._settings.contains("brain/url")
            if (has_saved_address or has_legacy_address) and self.address_input.text().strip():
                self.connect_to_brain()

    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setToolTip("")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _show_message(self, text):
        self.message_label.setText(text)
        self.content.setCurrentWidget(self.message_label)

    def _cancel_check(self):
        self._generation += 1
        if self._reply is not None:
            reply, self._reply = self._reply, None
            reply.abort()

    def _clear_dashboard(self):
        self._page_loaded = False
        self._page_loading = False
        if self._browser is not None:
            self._browser.stop()
            self._browser.setUrl(QUrl("about:blank"))

    def connect_to_brain(self):
        try:
            address = normalize_brain_url(self.address_input.text())
        except ValueError as error:
            self.address_error_label.setText(str(error))
            self.address_error_label.show()
            self.address_input.setToolTip(str(error))
            self.address_input.setFocus()
            self.address_input.selectAll()
            if not self._enabled:
                self._set_status(str(error), "#d97706")
            return

        self._cancel_check()
        self._clear_dashboard()
        self._base_url = address
        self._archive.set_source(address, False)
        self.address_input.setText(address)
        self.address_input.setToolTip("")
        self.address_error_label.hide()
        self._addresses[self._connection_mode] = address
        self._settings.setValue(f"brain/{self._connection_mode}_url", address)
        self._settings.setValue("brain/connection_mode", self._connection_mode)
        self._settings.setValue("brain/url", address)
        self._settings.sync()
        self._enabled = True
        self._online = False
        self.disconnect_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.last_seen_label.setText("Updated —")
        self.last_seen_label.setToolTip("Last successful sensor check")
        self._set_status("Connecting…", "#d97706")
        self._show_message("Connecting to Sentinel…")
        self._timer.start()
        self._check_connection()

    def disconnect_from_brain(self):
        self._enabled = False
        self._online = False
        self._timer.stop()
        self._cancel_check()
        self._clear_dashboard()
        self.disconnect_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self._set_status("Disconnected", "#d97706")
        self._show_message("Connect to view Sentinel.")
        self._archive.set_source(self._base_url, False)

    def _check_connection(self):
        if not self._enabled or self._reply is not None:
            return
        request = QNetworkRequest(QUrl(self._base_url + "/api/sensors"))
        request.setTransferTimeout(self.REQUEST_TIMEOUT_MS)
        request.setRawHeader(b"Accept", b"application/json")
        request.setAttribute(QNetworkRequest.Attribute.CacheLoadControlAttribute,
                             QNetworkRequest.CacheLoadControl.AlwaysNetwork)
        reply = self._network.get(request)
        self._reply = reply
        generation = self._generation
        reply.finished.connect(lambda: self._connection_checked(reply, generation))

    def _connection_checked(self, reply, generation):
        try:
            if generation != self._generation or not self._enabled:
                return
            if self._reply is reply:
                self._reply = None
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self._connection_failed(reply.errorString())
                return
            try:
                sensors = json.loads(bytes(reply.readAll()))
                if not isinstance(sensors, list) or any(
                    not isinstance(sensor, dict) or "id" not in sensor for sensor in sensors
                ):
                    raise ValueError("expected a sensor array")
            except (ValueError, UnicodeDecodeError) as error:
                self._connection_failed("Invalid /api/sensors response: " + str(error))
                return
            self._online = True
            self._archive.set_source(self._base_url, True)
            now = QDateTime.currentDateTime()
            self.last_seen_label.setText("Updated " + now.toString("HH:mm:ss"))
            self.last_seen_label.setToolTip("Last successful sensor check: " + now.toString("yyyy-MM-dd HH:mm:ss"))
            count = len(sensors)
            self._set_status(
                f"Connected · {count} {'sensor' if count == 1 else 'sensors'}",
                "#16a34a",
            )
            self.status_label.setToolTip("Service reachable. Sensor health is shown in the dashboard.")
            if not self._page_loaded and not self._page_loading:
                self._load_dashboard()
        finally:
            reply.deleteLater()

    def _connection_failed(self, reason):
        self._online = False
        self._archive.set_source(self._base_url, False)
        self._set_status("Offline · retrying…", "#dc2626")
        self.status_label.setToolTip(f"{self._base_url}\n{reason}\nRetrying every 5 seconds.")
        if self._connection_mode == "wifi":
            advice = "Check Pi Wi-Fi and app.py. Campus Wi-Fi may block device access."
        else:
            advice = "Check the cable, Pi address and app.py."
        self._show_message(
            f"Cannot reach {self._base_url}.\n" + advice
        )
        # Stop the page and video streams so old values cannot appear live.
        if self._page_loaded or self._page_loading:
            self._clear_dashboard()

    def _load_dashboard(self):
        if QWebEngineView is None:
            self._show_message(
                "Web view unavailable. Install PySide6 or open in browser."
            )
            return
        if self._browser is None:
            self._browser = QWebEngineView(self.content)
            self.content.addWidget(self._browser)
            # A separate, memory-only profile keeps Sentinel downloads scoped to this page.
            self._profile = QWebEngineProfile(self)
            self._browser.setPage(SentinelWebPage(self._profile, self._browser, self))
            self._profile.downloadRequested.connect(self._download_requested)
            self._browser.loadFinished.connect(self._dashboard_loaded)
        self._page_loading = True
        self._show_message("Loading Sentinel…")
        self._browser.load(QUrl(self._base_url + "/"))

    def _dashboard_loaded(self, success):
        if not self._page_loading or not self._online or self._browser.url().scheme() == "about":
            return
        self._page_loading = False
        self._page_loaded = success
        if success:
            self.content.setCurrentWidget(self._browser)
            self._push_archive_state()
        else:
            self._show_message(
                "Dashboard unavailable. Retrying…"
            )

    def refresh_dashboard(self):
        if self._enabled:
            self._cancel_check()
            self._clear_dashboard()
            self._show_message("Refreshing…")
            self._check_connection()

    def open_in_browser(self):
        try:
            address = normalize_brain_url(self.address_input.text())
        except ValueError as error:
            self.address_error_label.setText(str(error))
            self.address_error_label.show()
            self.address_input.setToolTip(str(error))
            self.address_input.setFocus()
            return
        self.address_error_label.hide()
        QDesktopServices.openUrl(QUrl(address + "/"))

    def _download_requested(self, download):
        directory = self._settings.value("brain/download_directory", "")
        if not directory:
            directory = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        suggested = Path(directory) / Path(download.downloadFileName()).name
        filename, _ = QFileDialog.getSaveFileName(self, "Save Sentinel download", str(suggested), "All files (*)")
        if not filename:
            download.cancel()
            return
        destination = Path(filename)
        download.setDownloadDirectory(str(destination.parent))
        download.setDownloadFileName(destination.name)
        self._settings.setValue("brain/download_directory", str(destination.parent))
        self._downloads.add(download)

        def updated():
            state = download.state()
            if state not in (
                download.DownloadState.DownloadCompleted,
                download.DownloadState.DownloadCancelled,
                download.DownloadState.DownloadInterrupted,
            ):
                return
            if state == download.DownloadState.DownloadCompleted:
                self.download_label.setText("Saved: " + str(destination))
            else:
                self.download_label.setText("Download stopped: " + download.interruptReasonString())
            self._downloads.discard(download)

        download.stateChanged.connect(updated)
        self.download_label.setText("Downloading: " + destination.name)
        download.accept()

    def _is_pi_page(self, url):
        origin = QUrl(self._base_url)
        return bool(self._base_url and url.scheme() == origin.scheme() and url.host() == origin.host()
                    and url.port(443 if url.scheme() == "https" else 80) == origin.port(443 if origin.scheme() == "https" else 80))

    def _push_archive_state(self, *_):
        if self._browser is not None and self._page_loaded and self._is_pi_page(self._browser.url()):
            state = json.dumps(self._archive.state(), ensure_ascii=True)
            self._browser.page().runJavaScript("if(window.updateDesktopArchive) window.updateDesktopArchive(" + state + ");")

    def show_log_settings(self):
        self._archive.configure(self)
        self._push_archive_state()

    def shutdown(self):
        """Stop PC-side work; never stop the remote Pi service."""
        self.disconnect_from_brain()
        self._archive_timer.stop()
        self._archive.shutdown()
        for download in list(self._downloads):
            download.cancel()


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    application = QApplication(sys.argv)
    window = BrainPage()
    window.setWindowTitle("Orchestrator — Sentinel")
    window.resize(1100, 800)
    application.aboutToQuit.connect(window.shutdown)
    window.show()
    sys.exit(application.exec())
