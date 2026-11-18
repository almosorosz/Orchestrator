import json
import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog

from experiment.adaptive_loop import default_setup, validate_setup
from gui.loop_setup_dialog import LoopSetupDialog
from gui.stylish_widgets import StatusBadge


APP = QApplication.instance() or QApplication([])
PRESETS = Path(__file__).resolve().parents[1] / "examples" / "configurations"


class LoopSetupDialogTests(unittest.TestCase):
    def test_existing_presets_round_trip(self):
        for path in PRESETS.glob("*_five_run_*.json"):
            with self.subTest(preset=path.name):
                preset = json.loads(path.read_text())
                environment = preset["environment"]
                setup = validate_setup(json.loads(environment["loop_setup"]))
                dialog = LoopSetupDialog(setup, environment["loop_backend"], preset["control_panel"]["devices"])
                try:
                    self.assertEqual(dialog.collect(), setup)
                finally:
                    dialog.close()

    def test_recording_overrides_and_support_round_trip(self):
        setup = default_setup()
        setup["channels"] = {"Temperature": "TEMP.temp_internal", "AS scale": ""}
        setup["process"] = {"SolvMass": 0.075, "Process note": "existing metadata"}
        setup["initial_processes"] = {"exp1": {"SeedMass": 0, "InitialConcentration": .05}}
        setup["support_profiles"] = {"valve_state": {"points": [[0, 10, 12], [0, 1, 0]], "step": True}}
        setup["seed_pump"] = {"rpm": 120, "grams_per_second": 0.0000025}
        setup["recording"].update(initial_concentration=0, seed_percentiles=[10, 20, 30])
        dialog = LoopSetupDialog(setup, "bo", {"TEMP": {"type": "huber"}})
        try:
            self.assertEqual(dialog.collect(), setup)
            self.assertGreaterEqual(dialog.channels["Temperature"].findText("TEMP.temp_internal"), 0)
        finally:
            dialog.close()

    def test_invalid_settings_stay_open_and_original_is_preserved(self):
        setup = default_setup()
        dialog = LoopSetupDialog(setup, "bo")
        try:
            dialog.max_followups.setValue(3)
            dialog.followup_start.setText("-1")
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
            self.assertIn("follow-up crystallization start", dialog.error.text())
            self.assertEqual(setup["max_followups"], 1)
            dialog.followup_start.setText("0")
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(dialog.setup["max_followups"], 3)
        finally:
            dialog.close()

    def test_completed_badge_is_green(self):
        badge = StatusBadge("Completed")
        self.assertIn("background:#15803d", badge.styleSheet())
        badge.set_status("Inactive")
        self.assertIn("background:#64748b", badge.styleSheet())

    def test_loop_error_wraps_in_settings(self):
        from gui.pages.settings_page import SettingsPage

        page = SettingsPage()
        try:
            error = "LOOP setup needs attention. Enable a BO target with positive weight."
            page.set_message(error=error)
            self.assertEqual(page.error_label.text(), error)
            self.assertTrue(page.error_label.wordWrap())
            self.assertGreater(page.error_label.heightForWidth(180), page.error_label.fontMetrics().height())
        finally:
            page.close()
