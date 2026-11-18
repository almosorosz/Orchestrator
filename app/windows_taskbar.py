"""Assign native HICONs to our own Windows taskbar window, independent of Qt."""
import ctypes
from ctypes import wintypes
import logging
import sys

from app.paths import WINDOWS_ICON_PATH

logger = logging.getLogger("orchestrator.application")
_user32 = None
_icons = {}  # Keep HICONs alive for the process lifetime while Windows uses them.
_taskbar_created = None


def _api():
    global _user32, _taskbar_created
    if _user32 is None:
        api = ctypes.WinDLL("user32", use_last_error=True)
        api.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                  ctypes.c_int, ctypes.c_int, wintypes.UINT]
        api.LoadImageW.restype = wintypes.HANDLE
        api.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        api.SendMessageW.restype = ctypes.c_ssize_t
        api.GetSystemMetrics.argtypes = [ctypes.c_int]
        api.GetSystemMetrics.restype = ctypes.c_int
        api.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
        api.RegisterWindowMessageW.restype = wintypes.UINT
        _taskbar_created = api.RegisterWindowMessageW("TaskbarButtonCreated")
        _user32 = api
    return _user32


def apply_native_icons(window):
    from PySide6.QtGui import QGuiApplication
    if sys.platform != "win32" or QGuiApplication.platformName() != "windows":
        return
    try:
        api = _api()
        hwnd = int(window.winId())
        for kind, metric in ((0, 49), (1, 11)):  # ICON_SMALL/SM_CXSMICON; ICON_BIG/SM_CXICON
            size = api.GetSystemMetrics(metric)
            if size not in _icons:
                handle = api.LoadImageW(None, str(WINDOWS_ICON_PATH), 1, size, size, 0x10)
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                _icons[size] = handle
            api.SendMessageW(hwnd, 0x80, kind, _icons[size])  # WM_SETICON
    except OSError as error:
        logger.warning("Native Windows taskbar icon could not be applied: %s", error)


def is_taskbar_created(message):
    if sys.platform != "win32" or _taskbar_created is None:
        return False
    msg = wintypes.MSG.from_address(int(message))
    return bool(_taskbar_created and msg.message == _taskbar_created)


def native_icon_handles(window):
    """Diagnostic for our own window: small and large WM_GETICON handles."""
    api = _api()
    return tuple(api.SendMessageW(int(window.winId()), 0x7F, kind, 0) for kind in (0, 1))
