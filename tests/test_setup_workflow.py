"""Guided setup navigation and readiness regressions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QWidget

from experiment.adaptive_loop import default_setup
from gui.pages.settings_page import SettingsPage
from gui.setup_checklist import SetupChecklist, SetupItem
from gui.setup_readiness import imed_readiness, loop_readiness

APP = QApplication.instance() or QApplication([])


class SetupWorkflowTests(unittest.TestCase):
    def test_load_actions_explain_the_session_lock_and_work_after_end_session(self):
        from MAIN_WINDOW import MainWindow

        class Dashboard(QWidget):
            execution_active_changed = Signal(bool)
            active = True

            def set_theme(self, _):
                pass

            def is_execution_active(self):
                return self.active

        window = MainWindow()
        panel = window.control_panel
        settings = window.settings_page
        dashboard = Dashboard()
        try:
            panel.experiment_window = dashboard
            window.embed_experiment_dashboard(dashboard)
            window.set_buttons_enabled(False)
            with patch("MAIN_WINDOW.QMessageBox.warning") as warning, \
                 patch("MAIN_WINDOW.QFileDialog.getOpenFileName", return_value=("", "")) as picker:
                self.assertTrue(settings.load_preset_button.isEnabled())
                settings.load_preset_button.click()
                self.assertIn("click Stop", warning.call_args.args[2])
                self.assertIn("End Session", settings.session_notice.text())
                picker.assert_not_called()

                dashboard.active = False
                dashboard.execution_active_changed.emit(False)
                self.assertIn("still open", settings.session_notice.text())
                panel.load_config_button.click()
                self.assertEqual(warning.call_args.args[1], "Cannot load configuration")
                self.assertIn("Go to Experiment and click End Session", warning.call_args.args[2])
                picker.assert_not_called()

                window.experiment_page.end_session_button.click()
                self.assertFalse(panel.experiment_window_attached())
                self.assertTrue(settings.session_notice.isHidden())
                settings.load_preset_button.click()
                panel.load_config_button.click()
                self.assertEqual(picker.call_count, 2)
        finally:
            dashboard.active = False
            window.end_experiment_session()
            window.close()

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

    def test_final_size_sample_count_follows_the_selected_modeling_mode(self):
        page = SettingsPage()
        try:
            page.loop_backend.setCurrentIndex(page.loop_backend.findData("imed"))
            count = page.pbm_final_size_samples
            self.assertEqual(count.value(), 1)
            self.assertFalse(count.isHidden())
            count.setValue(12)
            page.pbm_size_samples.setCurrentIndex(1)
            self.assertTrue(count.isHidden())
            self.assertEqual(page.final_size_sample_count(), 1)
            page.loop_backend.setCurrentIndex(page.loop_backend.findData("bo"))
            self.assertFalse(count.isHidden())
            self.assertEqual(page.final_size_sample_count(), 12)
            page.set_controls_enabled(False)
            self.assertFalse(count.isEnabled())
        finally:
            page.close()

    def test_automatic_setup_edits_shared_models_before_recording_experiments(self):
        from PySide6.QtWidgets import QDialog, QTabWidget
        from MAIN_WINDOW import MainWindow
        from imed_python.input_settings_dialogs import ObjectivesDialog, SolubilityDialog

        window = MainWindow()
        page = window.settings_page
        panel, bo = window.imed_page.panel, window.bo_page
        original_page = window.stack.currentWidget()
        try:
            page.loop_backend.setCurrentIndex(page.loop_backend.findData("imed"))
            editor = page.edit_loop_setup()

            def save_objectives(dialog):
                self.assertFalse(panel.case.get("ExpData"))
                dialog.accept()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
                dialog.fields[2].setValue(2.5)
                dialog.accept()
                return dialog.result()

            def save_solubility(dialog):
                dialog.coefficients.load([[-2.3], *([[0.]] * 7)])
                dialog.draw_plot()
                self.assertEqual(dialog.chart.series()[0].count(), 200)
                dialog.accept()
                return dialog.result()

            with patch.object(ObjectivesDialog, "exec", save_objectives):
                editor.navigate_requested.emit("objectives")
            with patch.object(SolubilityDialog, "exec", save_solubility):
                editor.navigate_requested.emit("solubility")
            self.assertEqual(panel.case["peweights"]["W3"], 2.5)
            self.assertEqual(panel.case["Solubility"]["a"], -2.3)
            self.assertFalse(panel.case.get("ExpData"))
            self.assertIs(window.stack.currentWidget(), original_page)
            editor.reject()

            page.loop_backend.setCurrentIndex(page.loop_backend.findData("bo"))
            editor = page.edit_loop_setup()
            bo.exploration.setText("1.5")

            def edit_bo(dialog):
                tabs = dialog.findChild(QTabWidget)
                self.assertEqual(tabs.tabText(0), "Input bounds and targets")
                self.assertTrue(tabs.isAncestorOf(bo.exploration))
                self.assertEqual(bo.exploration.text(), "1.5")
                bo.exploration.setText("2.5")
                return QDialog.DialogCode.Rejected

            with patch.object(QDialog, "exec", edit_bo):
                editor.navigate_requested.emit("bo_settings")
            self.assertTrue(bo.tabs.isAncestorOf(bo.exploration))
            self.assertEqual(bo.settings()["exploration"], 2.5)
            self.assertIs(window.stack.currentWidget(), original_page)
            editor.reject()
        finally:
            window.close()
            APP.processEvents()

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
        self.assertIn("Experiment period", checks["period"].detail)
        setup["mixing"]["enabled"] = True
        checks = {item.key: item for item in loop_readiness(setup, "imed", context)}
        self.assertFalse(checks["inputs"].ready)
        self.assertIn("stirrer", checks["inputs"].detail)

    def test_process_inputs_names_missing_densities_and_accepts_a_derived_solvent_mass(self):
        from experiment.models import Campaign, DeviceProfile, ExperimentDefinition, ProfilePoint

        temperature = DeviceProfile("TEMP", (ProfilePoint(0., 25.), ProfilePoint(20., 25.)))
        pump = DeviceProfile("P", (ProfilePoint(0., 60.), ProfilePoint(10., 0.), ProfilePoint(20., 0.)), "step")
        exp = ExperimentDefinition("exp1", (temperature, pump), model_window=(10., 20.))
        pump_config = {"type": "masterflex", "pump_calibration": {
            "version": 1, "input_unit": "g/h", "output_unit": "RPM", "points": [[0., 0.], [3600., 60.]]}}
        setup = default_setup()
        setup["recording"].update(initial_concentration=.09, no_feed=True)
        setup["process"].update(SolidDensity=1.2, ShapeFactor=.52)
        context = {"campaign": Campaign("DoE", (exp,)), "roles": {"T": "TEMP", "solution": "P"},
                   "configs": {"TEMP": {"type": "huber"}, "P": pump_config}, "connected": ["TEMP", "P"]}

        check = next(item for item in loop_readiness(setup, "imed", context) if item.key == "inputs")
        self.assertFalse(check.ready)
        self.assertIn("Solvent density (g/cm³)", check.detail)
        self.assertIn("Antisolvent density (g/cm³)", check.detail)
        setup["process"].update(SolventDensity=1., AntiSolventDensity=.9)
        check = next(item for item in loop_readiness(setup, "imed", context) if item.key == "inputs")
        self.assertTrue(check.ready, check.detail)

    def test_process_amounts_preview_edit_and_restore_per_run(self):
        from dataclasses import replace
        from experiment.models import Campaign, DeviceProfile, ExperimentDefinition, ProfilePoint
        from gui.loop_setup_dialog import LoopSetupDialog

        temperature = DeviceProfile("TEMP", (ProfilePoint(0., 25.), ProfilePoint(20., 25.)))
        solution = DeviceProfile("P", (ProfilePoint(0., 60.), ProfilePoint(10., 0.), ProfilePoint(20., 0.)), "step")
        seed = DeviceProfile("S", (ProfilePoint(0., 50.), ProfilePoint(10., 0.), ProfilePoint(20., 0.)), "step")
        exp = ExperimentDefinition("exp1", (temperature, solution, seed), model_window=(10., 20.))
        second = replace(exp, name="exp2", seed_mass_g=.3)
        campaign = Campaign("DoE", (exp, second), line_roles={"seed_lane": {"role": "seed", "name": "Seed"}})
        configs = {"TEMP": {"type": "huber"}, "P": {"type": "masterflex", "pump_calibration": {
            "version": 1, "input_unit": "g/h", "output_unit": "RPM", "points": [[0., 0.], [3600., 60.]]}},
            "S": {"type": "masterflex", "seed_calibration": {"version": 1, "rpm": 50., "points": [[0., 0.], [10., .2]]}}}
        context = {"campaign": campaign, "roles": {"T": "TEMP", "solution": "P", "seed_lane": "S"},
                   "configs": configs, "connected": configs, "monitors": (), "case": {}, "source": "doe", "label": "DoE",
                   "folders": ("", ""), "messages": [], "bo": None}
        setup = default_setup()
        setup["recording"].update(initial_concentration=.09, no_feed=True)
        dialog = LoopSetupDialog(setup, "bo", configs, preview=lambda *_: context)
        try:
            grid = dialog.process_values
            self.assertEqual(grid.item(0, 1).text(), "9.1")
            self.assertEqual(grid.item(2, 1).text(), "0.2")
            self.assertEqual(grid.item(3, 1).text(), "")
            self.assertEqual(dialog.collect()["initial_processes"], {})
            grid.setCurrentCell(0, 1)
            grid.item(0, 1).setText("8.5")
            dialog.refresh_checklist()
            self.assertEqual(dialog.collect()["initial_processes"], {"exp1": {"SolvMass": 8.5}})
            self.assertEqual(grid.item(0, 2).text(), "Starting experiment override")
            dialog.starting_experiment.setCurrentText("exp2")
            self.assertEqual(grid.item(0, 1).text(), "9.1")
            self.assertEqual(grid.item(2, 1).text(), "0.3")
            dialog.starting_experiment.setCurrentText("exp1")
            self.assertTrue(dialog.reset_starting_value.isEnabled())
            dialog.reset_starting_value.click()
            self.assertEqual(dialog.collect()["initial_processes"], {})
            self.assertEqual(grid.item(0, 1).text(), "9.1")
            grid.item(3, 1).setText("11.5")
            dialog.refresh_checklist()
            self.assertEqual(dialog.collect()["initial_processes"], {"exp1": {"InitialVolume_ml": 11.5}})
            self.assertEqual(grid.item(3, 2).text(), "Starting experiment override")
        finally:
            dialog.close()
