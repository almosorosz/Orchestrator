"""Shared application identity and eagerly loaded window/taskbar icons."""
import logging
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter

from app.paths import LOGO_PATH

APP_ID = "crystalloop.controlcenter.1.0"
_icon = None
logger = logging.getLogger("orchestrator.application")


def configure_windows_identity():
    """Keep Windows taskbar grouping consistent across launch methods."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_app_id.argtypes = [ctypes.c_wchar_p]
        set_app_id.restype = ctypes.c_long
        result = set_app_id(APP_ID)
        if result < 0:
            logger.warning("Windows application identity could not be set: %s", result)
    except (AttributeError, OSError) as error:
        logger.warning("Windows application identity could not be set: %s", error)


def configure_application_icon(application, window=None):
    """Set both the global default and explicit main-window icon.

    Keep pixmaps in memory so later Windows size requests do not depend on
    reopening a PNG in a synchronized folder.
    """
    global _icon
    if _icon is None:
        source = QPixmap(str(LOGO_PATH))
        if source.isNull():
            logger.warning("Application logo could not be loaded: %s", LOGO_PATH)
            return
        _icon = QIcon()
        for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
            scaled = source.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            # Windows expects square small/large icons. Preserve the artwork's
            # aspect ratio inside a transparent square at every taskbar size.
            square = QPixmap(size, size)
            square.fill(Qt.GlobalColor.transparent)
            painter = QPainter(square)
            painter.drawPixmap((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
            painter.end()
            _icon.addPixmap(square)
    application.setWindowIcon(_icon)
    if window is not None:
        window.setWindowIcon(_icon)
        # Frameless widgets can acquire/recreate their native QWindow after
        # construction. Apply to that handle too once it exists.
        handle = window.windowHandle()
        if handle is not None:
            handle.setIcon(_icon)
            from app.windows_taskbar import apply_native_icons
            apply_native_icons(window)
