"""Short real-Windows smoke check; opens no hardware connections."""
import os
import sys
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "windows"

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from app.windows_taskbar import native_icon_handles, _icons
from app.paths import DATA_DIR
from MAIN_WINDOW import MainWindow


def main():
    if sys.platform != "win32":
        raise RuntimeError("This diagnostic requires Windows")
    app = QApplication([])
    settings = {"enabled": True, "sql_enabled": False, "output_folder": str(DATA_DIR / "runs"), "remote_sql_url": ""}
    with patch("gui.recording_settings_dialog.load_recording_settings", return_value=settings):
        window = MainWindow()
    window.setWindowTitle("CrystaLoop — Taskbar icon verification")
    failed = []

    def check(label):
        handles = native_icon_handles(window)
        print(f"{label}: Windows small/large HICON = {handles}", flush=True)
        if not all(handle and handle in _icons.values() for handle in handles):
            failed.append(label)

    def restore():
        check("After taskbar creation")
        window.hide()
        window.show()
        QTimer.singleShot(300, lambda: check("After hide/show"))

    def finish():
        check("Final check")
        window.close()
        app.quit()

    window.show()
    QTimer.singleShot(1000, restore)
    QTimer.singleShot(5000, finish)
    app.exec()
    if failed:
        raise AssertionError(f"Native icon assignment failed: {failed}")
    print("PASS: Native Windows icons remain assigned after taskbar creation and hide/show.", flush=True)


if __name__ == "__main__":
    main()
