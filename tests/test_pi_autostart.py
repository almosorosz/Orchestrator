"""Verify boot configuration and installer orchestration without touching a host service."""

from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import types
import unittest
from unittest.mock import MagicMock, patch

PI_DIR = Path(__file__).resolve().parents[1] / "brain/pi"
spec = importlib.util.spec_from_file_location("sentinel_autostart_test", PI_DIR / "install_autostart.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class AutostartTests(unittest.TestCase):
    def test_conda_environment_is_activated_by_prefix_at_boot(self):
        unit = installer.render_service(
            "/home/pi/Monitor files", "pi", "/home/pi/miniforge/envs/envmo/bin/python",
            "/home/pi/miniforge/bin/conda", "/home/pi/miniforge/envs/envmo",
        )
        command = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
        self.assertIn('"/home/pi/miniforge/bin/conda" "run" "--no-capture-output" "--prefix"', command)
        self.assertIn('"/home/pi/miniforge/envs/envmo"', command)
        self.assertIn('"/home/pi/miniforge/envs/envmo/bin/python" "-u" "/home/pi/Monitor files/app.py"', command)
        self.assertIn("User=pi\n", unit)
        self.assertIn("WorkingDirectory=/home/pi/Monitor files\n", unit)
        self.assertIn("Environment=THERMAL_DEVICE=auto", unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("WantedBy=multi-user.target", unit)

    def test_plain_python_does_not_require_conda(self):
        unit = installer.render_service("/opt/sentinel", "pi", "/opt/venv/bin/python")
        self.assertIn('ExecStart="/opt/venv/bin/python" "-u" "/opt/sentinel/app.py"', unit)
        self.assertNotIn('"run"', unit)

    def test_service_paths_preserve_spaces_and_literal_specifiers(self):
        unit = installer.render_service('/home/pi/100% $lab "A"', "pi", "/usr/bin/python3")
        self.assertIn('WorkingDirectory=/home/pi/100%% $lab "A"\n', unit)
        self.assertIn('"/home/pi/100%% $$lab \\"A\\"/app.py"', unit)
        for directory in ("relative", "/home/pi/test\nUser=root", "/home/pi/test\r", "/home/pi/test\\"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                installer.render_service(directory, "pi", "/usr/bin/python3")

    def test_missing_conda_executable_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Activate"):
            installer.render_service("/home/pi/sentinel", "pi", "/env/bin/python", conda_prefix="/env")

    def test_manual_server_port_conflict_is_not_silently_installed(self):
        with patch.object(installer.subprocess, "run", return_value=subprocess.CompletedProcess([], 3)), \
                patch.object(installer.socket, "socket") as create_socket:
            create_socket.return_value.__enter__.return_value.bind.side_effect = OSError("Address in use")
            with self.assertRaisesRegex(RuntimeError, "Ctrl\\+C"):
                installer.check_port()

    def test_existing_service_can_be_updated_without_port_probe(self):
        with patch.object(installer.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(installer.socket, "socket") as create_socket:
            installer.check_port()
        create_socket.assert_not_called()

    def test_ready_api_does_not_require_camera_to_be_plugged_in(self):
        response = io.BytesIO(json.dumps({"model": "TOPDON TC001", "enabled": True, "available": False}).encode())
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = response
        with patch.object(installer.urllib.request, "build_opener", return_value=opener):
            installer.wait_for_service(timeout=1)

    def test_disabled_or_old_api_does_not_report_success(self):
        response = io.BytesIO(json.dumps({"model": "TOPDON TC001", "enabled": False}).encode())
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = response
        with patch.object(installer.urllib.request, "build_opener", return_value=opener), \
                patch.object(installer.time, "monotonic", side_effect=[0, 0.1, 2]), \
                patch.object(installer.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "API did not become ready"):
                installer.wait_for_service(timeout=1)

    def main_context(self, *, dry_run=False):
        stack = ExitStack()
        stack.enter_context(patch.object(installer, "APP_DIR", PurePosixPath("/home/pi/Monitor files")))
        stack.enter_context(patch.object(installer.sys, "argv", ["install_autostart.py"] + (["--dry-run"] if dry_run else [])))
        stack.enter_context(patch.object(installer.sys, "platform", "linux"))
        stack.enter_context(patch.dict("sys.modules", {"pwd": types.SimpleNamespace(getpwuid=lambda uid: types.SimpleNamespace(pw_name="pi"))}))
        stack.enter_context(patch.object(installer.os, "geteuid", return_value=1000, create=True))
        stack.enter_context(patch.object(installer.Path, "is_dir", autospec=True,
                                        side_effect=lambda path: str(path).replace("\\", "/") == "/run/systemd/system"))
        return stack

    def test_install_enables_and_restarts_service_only_after_preflight(self):
        with self.main_context(), patch.object(installer, "verify_environment") as verify, \
                patch.object(installer, "check_port") as check_port, \
                patch.object(installer, "wait_for_service") as ready, \
                patch.object(installer.subprocess, "run") as run, \
                patch.object(installer.sys, "stdout", new_callable=io.StringIO):
            self.assertEqual(installer.main(), 0)
        verify.assert_called_once()
        check_port.assert_called_once()
        ready.assert_called_once()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[-3:], [
            ["sudo", "systemctl", "daemon-reload"],
            ["sudo", "systemctl", "enable", "sentinel.service"],
            ["sudo", "systemctl", "restart", "sentinel.service"],
        ])

    def test_preflight_failure_leaves_services_unchanged(self):
        with self.main_context(), \
                patch.object(installer, "verify_environment", side_effect=RuntimeError("Missing dependencies")), \
                patch.object(installer.subprocess, "run") as run, \
                patch.object(installer.sys, "stderr", new_callable=io.StringIO):
            self.assertEqual(installer.main(), 1)
        run.assert_not_called()

    def test_dry_run_never_requests_sudo_or_installs_service(self):
        with self.main_context(dry_run=True), patch.object(installer, "verify_environment") as verify, \
                patch.object(installer.subprocess, "run") as run, \
                patch.object(installer.sys, "stdout", new_callable=io.StringIO) as output:
            self.assertEqual(installer.main(), 0)
        self.assertIn("[Service]", output.getvalue())
        verify.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
