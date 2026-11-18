import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


class PackagingTests(unittest.TestCase):
    def test_startup_failure_produces_diagnostic_without_opening_ui(self):
        import launcher
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            output = Path(root) / "startup.json"
            with patch.object(sys, "argv", ["CrystaLoop.exe", "--smoke-test", str(output)]), \
                 patch.object(launcher.runpy, "run_module", side_effect=RuntimeError("missing packaged resource")):
                self.assertEqual(launcher.main(), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertIn("missing packaged resource", report["error"])

    def test_frozen_writable_data_is_separate_from_bundled_resources(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as local:
            with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", str(root), create=True), \
                 patch.dict(os.environ, {"LOCALAPPDATA": local, "CRYSTALOOP_DATA_DIR": ""}):
                paths = runpy.run_path(str(root / "app" / "paths.py"))
            self.assertEqual(paths["DATA_DIR"], Path(local) / "CrystaLoop" / "data")
            self.assertEqual(paths["PUMP_CALIBRATION_DIR"], root / "data" / "pump_calibration")
            self.assertEqual(paths["ASSETS_DIR"], root / "assets")
