"""Guided setup navigation and readiness regressions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PySide6.QtWidgets import QApplication

from experiment.adaptive_loop import default_setup
from gui.pages.settings_page import SettingsPage
from gui.setup_checklist import SetupChecklist, SetupItem
from gui.setup_readiness import imed_readiness, loop_readiness

APP = QApplication.instance() or QApplication([])


class SetupWorkflowTests(unittest.TestCase):
    def test_loop_draft_survives_navigation_and_only_save_applies_it(self):
        page = SettingsPage()
        page.loop_backend.setCurrentIndex(page.loop_backend.findData("bo"))
        try:
            dialog = page.edit_loop_setup()
            dialog.max_followups.setValue(3)
            dialog.navigate_requested.emit("folders")
            self.assertEqual(page.workflow_tabs.currentIndex(), 3)
            self.assertIs(page.edit_loop_setup(), dialog)
            self.assertEqual(dialog.max_followups.value(), 3)
            self.assertEqual(page.loop_setup["max_followups"], 1)
            dialog.reject()
            self.assertIsNone(page.loop_setup_dialog)
            dialog = page.edit_loop_setup()
            dialog.max_followups.setValue(2)
            dialog.accept()
            self.assertEqual(page.loop_setup["max_followups"], 2)
        finally:
            if page.loop_setup_dialog:
                page.loop_setup_dialog.reject()
            page.close()

    def test_checklist_changes_state_and_navigates_without_marking_missing_values_ready(self):
        checks = SetupChecklist()
        selected = []
        checks.activated.connect(selected.append)
        checks.set_items([SetupItem("a", "Inputs", False, "Enter masses"),
                          SetupItem("b", "Optional", False, "Optional overrides", required=False)])
        checks.next_button.click()
        self.assertEqual(selected, ["a"])
        self.assertFalse(checks.buttons["a"].property("ready"))
        checks.set_items([SetupItem("a", "Inputs", True, "Derived from charging stages")])
        self.assertTrue(checks.buttons["a"].property("ready"))
        self.assertIn("#15803d", checks.buttons["a"].styleSheet())
        self.assertFalse(checks.next_button.isEnabled())
        checks.close()

    def test_loop_draft_menus_refresh_new_devices_without_losing_saved_choices(self):
        from gui.loop_setup_dialog import LoopSetupDialog
        setup = default_setup()
        setup["channels"] = {"Temperature": "OLD.temp_internal"}
        dialog = LoopSetupDialog(setup, "bo")
        try:
            dialog._refresh_device_choices({"NEW": {"type": "huber"}})
            self.assertGreaterEqual(dialog.channels["Temperature"].findText("NEW.temp_internal"), 0)
            self.assertEqual(dialog.channels["Temperature"].currentText(), "OLD.temp_internal")
            self.assertGreaterEqual(dialog.assignments["T"].findText("NEW"), 0)
        finally:
            dialog.close()

    def test_settings_preserves_controls_in_four_full_width_tabs(self):
        page = SettingsPage()
        try:
            self.assertEqual(page.workflow_tabs.count(), 4)
            for widget in (page.profile_button, page.generate_button, page.assign_doe_button,
                           page.start_independent_doe_button, page.recording_button, page.reservoirs_button,
                           page.save_preset_button, page.load_preset_button, page.loop_setup_button,
                           page.pbm_size_samples, page.exp_path_input, page.MODEL_path_input):
                self.assertTrue(page.isAncestorOf(widget))
            with TemporaryDirectory(dir=Path.cwd()) as folder:
                page.exp_path_input.setText(folder)
                page.MODEL_path_input.setText(folder)
                self.assertTrue(page.workflow_checklist.buttons["files"].property("ready"))
                page.MODEL_path_input.clear()
                self.assertFalse(page.workflow_checklist.buttons["files"].property("ready"))
            page.set_doe_available(True)
            page.set_assignment_available(True)
            self.assertTrue(page.start_independent_doe_button.isEnabled())
            page.set_controls_enabled(False)
            self.assertFalse(page.start_independent_doe_button.isEnabled())
        finally:
            page.close()

    def test_timeline_preview_uses_readable_factor_names_and_retains_profiles(self):
        from experiment.experiment_service import ExperimentService
        from test_recipe_editor import template
        service = ExperimentService()
        service.exp_par.timeline = template()
        service.exp_par.v_heat_max = service.exp_par.v_cool_max = 1000
        creator = service.generate_initial_doe(3, "LHS")
        page = SettingsPage()
        try:
            page.render_doe_plots(creator, True)
            self.assertEqual(page.experiment_choice.count(), 3)
            self.assertIsNotNone(page.profile_canvas)
            self.assertGreater(page.variable_table.rowCount(), 0)
            normalized = creator.doe_sampler.scaler(creator.DoE_samples)
            if normalized.shape[1]:
                from scipy.spatial.distance import pdist
                distances = pdist(normalized[:, :1])
                self.assertEqual(page.sampling_summary.item(0, 1).text(), f"{distances.min():.3f}")
                self.assertEqual(page.sampling_summary.item(0, 2).text(), f"{distances.mean():.3f}")
            self.assertTrue(all("stage" not in choice.itemText(i).split(".")[0].lower() or "→" in choice.itemText(i)
                                for choice in page.factor_choices for i in range(choice.count())))
        finally:
            page.close()

    def test_single_legacy_experiment_can_be_previewed_without_sampled_factors(self):
        from experiment.experiment_service import ExperimentService
        creator = ExperimentService().generate_initial_doe(1, "LHS")
        page = SettingsPage()
        try:
            page.render_doe_plots(creator, True)
            self.assertEqual(page.experiment_choice.count(), 1)
            self.assertIsNotNone(page.profile_canvas)
            self.assertEqual(page.sampling_summary.rowCount(), 0)
        finally:
            page.close()

    def test_legacy_component_names_have_units_and_fixed_bounds_are_hidden(self):
        from experiment.profiles.profile_gen import ExperimentProfileParameters
        page = SettingsPage()
        try:
            page.show_definition(ExperimentProfileParameters())
            values = [[page.variable_table.item(row, col).text() for col in range(6)]
                      for row in range(page.variable_table.rowCount())]
            stirring = next(row for row in values if row[0] == "Main stirring rate")
            self.assertEqual(stirring[1:], ["Fixed", "350", "—", "—", "RPM"])
            self.assertEqual(page._factor_labels["T_final"], "Final temperature (°C)")
        finally:
            page.close()

    def test_imed_readiness_reports_inputs_and_optional_fit_separately(self):
        from imed_python.test_probabilistic import ProbabilisticTests
        empty = {item.key: item for item in imed_readiness({})}
        self.assertFalse(empty["experiments"].ready)
        self.assertFalse(empty["fits"].required)
        case = ProbabilisticTests.case()
        case["peweights"] = {"W1": 0, "W2": 1, "W3": 0}
        checks = {item.key: item for item in imed_readiness(case)}
        self.assertTrue(checks["models"].ready)
        self.assertTrue(checks["solubility"].ready)
        self.assertTrue(checks["pe"].ready)
        case["Solubility"] = {}
        self.assertFalse(next(item for item in imed_readiness(case) if item.key == "solubility").ready)

    def test_loop_does_not_mark_missing_monitors_or_targets_green(self):
        from test_run_stages import manual_campaign, CONFIGS
        setup = default_setup()
        setup["recording"].update(no_feed=True, initial_concentration=.1)
        from bo.settings import empty_settings
        bo = empty_settings()
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            context = {"campaign": manual_campaign(), "roles": {"T": "TEMP"}, "configs": CONFIGS,
                       "monitors": ["missing_probe"], "connected": ["TEMP"], "case": {},
                       "folders": (folder, folder), "bo": bo}
            checks = {item.key: item for item in loop_readiness(setup, "bo", context)}
            self.assertFalse(checks["devices"].ready)
            self.assertIn("missing_probe", checks["devices"].detail)
            self.assertFalse(checks["model"].ready)
            bo["targets"]["d50"].update(enabled=True, weight=1)
            checks = {item.key: item for item in loop_readiness(setup, "bo", context)}
            self.assertTrue(checks["model"].ready)
            self.assertFalse(checks["recording"].ready)

    def test_legacy_profiles_use_recording_offsets_without_requiring_named_stages(self):
        from dataclasses import replace
        from test_run_stages import manual_campaign, CONFIGS
        campaign = manual_campaign()
        campaign = replace(campaign, experiments=tuple(replace(exp, run_stages=(), model_window=None)
                                                       for exp in campaign.experiments))
        setup = default_setup()
        setup["recording"].update(no_feed=True, initial_concentration=.1)
        context = {"campaign": campaign, "roles": {"T": "TEMP"}, "configs": CONFIGS}
        checks = {item.key: item for item in loop_readiness(setup, "imed", context)}
        self.assertTrue(checks["period"].ready)
        self.assertIn("Recording", checks["period"].detail)
        setup["mixing"]["enabled"] = True
        checks = {item.key: item for item in loop_readiness(setup, "imed", context)}
        self.assertFalse(checks["inputs"].ready)
        self.assertIn("stirrer", checks["inputs"].detail)
