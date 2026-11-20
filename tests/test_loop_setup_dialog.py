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

    def test_manual_times_are_only_editable_on_the_experiment_period_tab(self):
        setup = default_setup()
        setup["recording"].update(start_s=2, end_s=12)
        dialog = LoopSetupDialog(setup, "bo")
        try:
            self.assertTrue(dialog.period.isAncestorOf(dialog.manual_start))
            self.assertTrue(dialog.period.manual_form.isHidden())
            self.assertEqual(dialog.collect()["recording"]["start_s"], 2)

            dialog.period.mode.setCurrentIndex(dialog.period.mode.findData("manual"))
            self.assertFalse(dialog.period.manual_form.isHidden())
            dialog.manual_start.setText("4")
            dialog.manual_end.clear()
            recording = dialog.collect()["recording"]
            self.assertEqual(recording["start_s"], 4)
            self.assertIsNone(recording["end_s"])

            dialog.manual_end.setText("3")
            with self.assertRaisesRegex(ValueError, "end must follow the start"):
                dialog.collect()
            dialog.period.mode.setCurrentIndex(dialog.period.mode.findData("definition"))
            self.assertTrue(dialog.period.manual_form.isHidden())
            self.assertEqual(dialog.collect()["recording"]["end_s"], 12)
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

    def test_unknown_saved_fields_can_be_reviewed_and_removed_without_resetting_settings(self):
        from gui.pages.settings_page import SettingsPage

        page = SettingsPage()
        page.loop_backend.setCurrentIndex(page.loop_backend.findData("bo"))
        page.loop_setup["max_followups"] = 6
        page.loop_setup["obsolete_setting"] = {"value": 21}
        try:
            with self.assertRaisesRegex(ValueError, "obsolete_setting.*Open Automatic loop setup"):
                validate_setup(page.loop_setup)
            dialog = page.edit_loop_setup()
            self.assertIn("obsolete_setting", dialog.compatibility_note.text())
            self.assertEqual(dialog.max_followups.value(), 6)
            dialog.reject()
            self.assertIn("obsolete_setting", page.loop_setup)

            dialog = page.edit_loop_setup()
            dialog.accept()
            self.assertNotIn("obsolete_setting", page.loop_setup)
            self.assertEqual(validate_setup(page.loop_setup)["max_followups"], 6)
            with self.assertRaisesRegex(ValueError, "control_panel"):
                LoopSetupDialog({"control_panel": {}}, "bo")
        finally:
            page.close()

    def test_running_app_with_old_loop_module_gets_restart_message(self):
        from unittest.mock import patch
        from gui.pages.settings_page import SettingsPage

        page = SettingsPage()
        page.loop_backend.setCurrentIndex(page.loop_backend.findData("bo"))
        try:
            with patch("experiment.adaptive_loop.setup_for_editing", None), patch("gui.pages.settings_page.QMessageBox.warning") as warn:
                self.assertIsNone(page.edit_loop_setup())
            self.assertIn("Close and reopen the app", warn.call_args.args[2])
        finally:
            page.close()

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
