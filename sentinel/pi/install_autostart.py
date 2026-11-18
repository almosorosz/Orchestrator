"""Install the Sentinel boot service from its working Python/conda environment.

Run as your normal Pi user: python install_autostart.py
Preview without installing: python install_autostart.py --dry-run
"""

import argparse
import importlib
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import json


SERVICE = "sentinel.service"
APP_DIR = Path(__file__).resolve().parent


def unit_quote(value, *, command=False):
    """Quote one systemd value, including literal specifiers and command dollars."""
    value = str(value)
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Service paths must not contain control characters")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def unit_path(value):
    # WorkingDirectory is a scalar path: unlike ExecStart it does not strip quotes.
    value = str(value)
    if not value.startswith("/") or value != value.strip() or value.endswith("\\"):
        raise ValueError("The service directory must be an absolute Linux path without trailing whitespace or backslash")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Service paths must not contain control characters")
    return value.replace("%", "%%")


def render_service(app_dir, username, python, conda_exe=None, conda_prefix=None):
    """Use the selected environment without relying on an interactive login shell."""
    app_dir = str(app_dir)
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.-]*\$?", username):
        raise ValueError("Invalid Pi username")
    app_path = app_dir.rstrip("/") + "/app.py"
    command = [python, "-u", app_path]
    if conda_prefix:
        if not conda_exe:
            raise ValueError("Activate the conda environment before installing autostart")
        command = [conda_exe, "run", "--no-capture-output", "--prefix", conda_prefix, *command]
    return "\n".join([
        "[Unit]",
        "Description=Sentinel Pi monitoring service",
        "After=network.target",
        "StartLimitIntervalSec=0",
        "",
        "[Service]",
        "Type=simple",
        "User=" + username,
        "WorkingDirectory=" + unit_path(app_dir),
        "Environment=PYTHONUNBUFFERED=1",
        "Environment=THERMAL_DEVICE=auto",
        "ExecStart=" + " ".join(unit_quote(arg, command=True) for arg in command),
        "Restart=on-failure",
        "RestartSec=5",
        "TimeoutStopSec=15",
        "StandardOutput=journal",
        "StandardError=journal",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ])


def verify_environment():
    for module in ("flask", "serial", "numpy", "cv2"):
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError(
                "Missing camera dependencies in this Python environment. Run: "
                "conda install -c conda-forge flask pyserial numpy opencv"
            ) from error
    for name in ("app.py", "history_store.py", "tc001_reader.py", "pico_serial_reader.py", "fumehood_dashboard.html"):
        if not (APP_DIR / name).is_file():
            raise RuntimeError("Missing deployment file: " + name)


def check_port():
    """An existing service may be updated; a manual process must be stopped first."""
    active = subprocess.run(
        ["systemctl", "is-active", "--quiet", SERVICE], check=False
    ).returncode == 0
    if active:
        return
    try:
        with socket.socket() as listener:
            listener.bind(("0.0.0.0", 8000))
    except OSError as error:
        raise RuntimeError(
            "Port 8000 is already in use. Stop the manually running app.py with "
            "Ctrl+C, then run this installer again."
        ) from error


def wait_for_service(timeout=30):
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open("http://127.0.0.1:8000/api/thermal", timeout=1) as response:
                data = json.load(response)
            if data.get("model") == "TOPDON TC001" and data.get("enabled"):
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError(
        "The service was installed, but its API did not become ready. "
        "Check: journalctl -u sentinel.service -n 50 --no-pager"
    )


def main():
    parser = argparse.ArgumentParser(description="Enable Sentinel at Pi boot using the current Python environment")
    parser.add_argument("--dry-run", action="store_true", help="Print the service configuration without installing")
    args = parser.parse_args()
    try:
        if sys.platform != "linux" or not Path("/run/systemd/system").is_dir():
            raise RuntimeError("Run this installer on the Pi, using a Linux system with systemd")
        import pwd
        username = pwd.getpwuid(os.geteuid()).pw_name
        conda_prefix = str(Path(sys.prefix)) if (Path(sys.prefix) / "conda-meta").is_dir() else None
        conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
        service = render_service(APP_DIR, username, sys.executable, conda_exe, conda_prefix)
        if args.dry_run:
            print(service, end="")
            return 0
        if os.geteuid() == 0:
            raise RuntimeError("Run as your normal Pi user without sudo; the installer requests sudo when needed")
        verify_environment()
        if conda_prefix and (not conda_exe or not Path(conda_exe).is_file()):
            raise RuntimeError("Cannot find conda. Activate your working conda environment and try again")
        check_port()
        print("Installing Sentinel boot service using: " + sys.executable, flush=True)
        subprocess.run(["sudo", "-v"], check=True)
        with tempfile.TemporaryDirectory(prefix="sentinel-autostart-") as directory:
            source = Path(directory) / SERVICE
            source.write_text(service, encoding="utf-8")
            subprocess.run(["sudo", "install", "-m", "0644", str(source), "/etc/systemd/system/" + SERVICE], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", SERVICE], check=True)
        subprocess.run(["sudo", "systemctl", "restart", SERVICE], check=True)
        wait_for_service()
        print("Sentinel is running and will start automatically at boot.")
        print("TC001 detection is enabled; check /api/thermal for camera status.")
        print("Status: systemctl status sentinel.service --no-pager")
        print("Logs: journalctl -u sentinel.service -f")
        print("Disable autostart: sudo systemctl disable --now sentinel.service")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("[autostart] " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
