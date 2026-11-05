import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from experiment.doe_campaign import DoeCampaignService, ProfileAssignment, DEFAULT_ASSIGNMENTS
from experiment.models import RunStatus
from experiment_dashboard.experiment_dashboard import ExperimentWorker
from control_panel.control_panel import DEFAULT_DEVICES, HARDWARE_TYPES, create_hardware
from gui.doe_assignment_dialog import DoeAssignmentDialog

APP = QApplication.instance() or QApplication([])


def fixture_creator():
    frame = pd.DataFrame({
        "t": [0., .0003], "T": [20., 25.], "AS": [60., 0.], "solution": [10., 0.],
        "solvent": [0., 0.], "clean_sol": [0., 0.], "seed": [0., 50.],
        "stirring_rate": [100., 200.], "stirring_rate_seed": [150., 200.],
        "stirring_rate_sol": [200., 250.], "valve_state": [0., 1.],
    })
    converter = SimpleNamespace(convert_to_rpm=lambda values: np.asarray(values) * 2,
                                res=SimpleNamespace(x=np.array([0., 2.])), folder_loc="fixture_calibration.csv")
    storage = SimpleNamespace(data={"exp1": frame, "exp2": frame.copy()},
                              DoE={"exp1": pd.DataFrame({"seed": [1.]}), "exp2": pd.DataFrame({"seed": [2.]})},
                              as_rate_converter=converter, solution_rate_converter=converter,
                              solvent_rate_converter=converter, cleaning_sol_rate_converter=converter)
    return SimpleNamespace(exp_storage=storage, TYPE="TEST", parameter_snapshots={
        "exp1": {"seed": {"value": 1.}, "T_room": 20.},
        "exp2": {"seed": {"value": 2.}, "T_room": 20.},
    })


def fixture_configs():
    configs = deepcopy(DEFAULT_DEVICES)
    for config in configs.values():
        config["simulated"] = True
    return configs


def fixture_assignments():
    return tuple(ProfileAssignment(role, device_id, .8 if role == "AS" else None)
                 for role, device_id in DEFAULT_ASSIGNMENTS.items() if role not in {"solvent", "clean_sol"})


class CompilationTests(unittest.TestCase):
    def setUp(self):
        self.creator = fixture_creator()
        self.configs = fixture_configs()
        self.service = DoeCampaignService(self.creator)
        self.assignments = fixture_assignments()

    def test_hours_density_calibration_and_step_interpolation(self):
        campaign = self.service.compile(self.assignments, self.configs)
        self.assertEqual(len(campaign.experiments), 2)
        experiment = campaign.experiments[0]
        profiles = {p.device_id: p for p in experiment.profiles}
        self.assertAlmostEqual(profiles["TEM_MAIN"].points[-1].time_s, 1.08)
        self.assertAlmostEqual(profiles["PMP_AS"].points[0].target, 1.)
        self.assertEqual(profiles["PMP_AS"].unit, "mL/min")
        self.assertEqual(profiles["PMP_SOLU"].points[0].target, 20.)
        self.assertEqual(profiles["PMP_SEED"].points[-1].target, 50.)
        self.assertEqual(profiles["VALVE"].interpolation, "step")
        self.assertEqual(experiment.source, "doe")

    def test_fixed_density_needs_no_input_and_ignores_legacy_values(self):
        for value in (None, .8, 1.2, 0, -1, float("nan")):
            assignments = tuple(replace(a, density_g_ml=value) if a.role == "AS" else a for a in self.assignments)
            with self.subTest(value=value):
                campaign = self.service.compile(assignments, self.configs)
                profile = next(p for p in campaign.experiments[0].profiles if p.device_id == "PMP_AS")
                self.assertAlmostEqual(profile.points[0].target, 1.)

    def test_missing_duplicate_and_incompatible_assignments_rejected(self):
        with self.assertRaisesRegex(ValueError, "Assign active"):
            self.service.compile(self.assignments[1:], self.configs)
        assignments = tuple(replace(a, device_id="TEM_MAIN") if a.role == "stirring_rate" else a for a in self.assignments)
        with self.assertRaisesRegex(ValueError, "different device"):
            self.service.compile(assignments, self.configs)
        configs = deepcopy(self.configs)
        configs["TEM_MAIN"]["type"] = "jkem"
        with self.assertRaisesRegex(ValueError, "cannot be assigned"):
            self.service.compile(self.assignments, configs)

    def test_device_limits_apply_after_conversion(self):
        self.creator.exp_storage.data["exp1"]["solution"] = [200., 0.]
        service = DoeCampaignService(self.creator)
        with self.assertRaisesRegex(ValueError, "PMP_SOLU"):
            service.compile(self.assignments, self.configs)

    def test_bad_timeline_and_invalid_monitor_rejected(self):
        self.creator.exp_storage.data["exp1"]["t"] = [0., 0.]
        with self.assertRaisesRegex(ValueError, "times"):
            DoeCampaignService(self.creator).compile(self.assignments, self.configs)
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.service.compile(self.assignments, self.configs, ["TEM_MAIN"])

    def test_identical_duplicate_boundary_rows_are_coalesced(self):
        frame = self.creator.exp_storage.data["exp1"]
        self.creator.exp_storage.data["exp1"] = pd.concat([frame.iloc[:1], frame], ignore_index=True)
        campaign = DoeCampaignService(self.creator).compile(self.assignments, self.configs)
        self.assertEqual(len(campaign.experiments[0].profiles[0].points), 2)

    def test_source_data_and_compiled_campaign_do_not_share_mutable_frames(self):
        campaign = self.service.compile(self.assignments, self.configs)
        self.creator.exp_storage.data["exp1"]["T"] = [100, 100]
        self.assertEqual(campaign.experiments[0].profiles[0].points[0].target, 20)
        self.assertEqual(self.service.frames["exp1"]["T"].iloc[0], 20)

    def test_monitoring_and_campaign_identity_reach_executor(self):
        campaign = self.service.compile(self.assignments, self.configs, ["SCL_AS"])
        selected = {p.device_id for e in campaign.experiments for p in e.profiles} | set(campaign.monitor_device_ids)
        configs = {key: self.configs[key] for key in selected}
        hardware = create_hardware(configs, HARDWARE_TYPES)
        for device in hardware.values():
            device.connect()
        worker = ExperimentWorker(hardware, configs, campaign.experiments, campaign=campaign,
                                  speed=100, end_hold_s=0, poll_interval=.01)
        worker.start_experiment()
        self.assertEqual(worker.outcome, RunStatus.COMPLETED)
        self.assertEqual(worker.campaign.campaign_id, campaign.campaign_id)
        self.assertEqual([r.experiment_id for r in worker.runs], [e.experiment_id for e in campaign.experiments])

    def test_real_generated_doe_converts_with_existing_calibrations(self):
        from experiment.experiment_service import ExperimentService
        creator = ExperimentService().generate_initial_doe(2, "GREEDY")
        service = DoeCampaignService(creator)
        assignments = tuple(ProfileAssignment(role, DEFAULT_ASSIGNMENTS[role], .8 if role == "AS" else None)
                            for role in DEFAULT_ASSIGNMENTS if role in service.required_roles())
        campaign = service.compile(assignments, self.configs)
        self.assertEqual(len(campaign.experiments), 2)
        self.assertGreater(campaign.experiments[0].profiles[0].points[-1].time_s, 3600)
        self.assertEqual(len(creator.parameter_snapshots), 2)


class CsvTests(unittest.TestCase):
    def test_raw_and_assigned_csv_are_inspectable_and_never_overwrite(self):
        service = DoeCampaignService(fixture_creator())
        configs, assignments = fixture_configs(), fixture_assignments()
        campaign = service.compile(assignments, configs)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as parent:
            raw = service.export_csv(parent)
            self.assertTrue({"parameters.csv", "generated_profiles.csv", "metadata.json", "exp1_generated.csv", "exp2_generated.csv"}.issubset(p.name for p in raw.iterdir()))
            parameters = pd.read_csv(raw / "parameters.csv")
            self.assertEqual(parameters["seed"].tolist(), [1., 2.])
            profiles = pd.read_csv(raw / "generated_profiles.csv")
            self.assertAlmostEqual(profiles["time_s"].max(), 1.08)
            self.assertIn("T", profiles.columns)
            self.assertIn("AS", profiles.columns)
            self.assertNotIn("profile_role", profiles.columns)
            assigned = service.export_csv(parent, campaign, assignments, configs)
            self.assertNotEqual(raw, assigned)
            targets = pd.read_csv(assigned / "assigned_profiles.csv")
            self.assertEqual(set(targets["campaign_id"]), {campaign.campaign_id})
            self.assertEqual(set(targets["experiment_id"]), {e.experiment_id for e in campaign.experiments})
            self.assertIn("TEM_MAIN", targets.columns)
            self.assertEqual(list(pd.read_csv(assigned / "exp1_assigned.csv").columns).count("time_s"), 1)
            metadata = json.loads((assigned / "metadata.json").read_text())
            solution = next(a for a in metadata["assignments"] if a["role"] == "solution")
            self.assertEqual(solution["calibration_coefficients"], [0., 2.])
            self.assertTrue((raw / "parameters.csv").exists())


class AssignmentGuiTests(unittest.TestCase):
    def setUp(self):
        settings = {"enabled": True, "sql_enabled": False, "output_folder": str(Path.cwd() / "data" / "runs"), "remote_sql_url": ""}
        mocked = patch("gui.recording_settings_dialog.load_recording_settings", return_value=settings)
        mocked.start()
        self.addCleanup(mocked.stop)
    def wait_for(self, predicate, timeout=15):
        end = time.monotonic() + timeout
        while not predicate() and time.monotonic() < end:
            APP.processEvents()
            time.sleep(.01)
        self.assertTrue(predicate())

    def test_assignment_applies_without_start_and_routes_are_separate(self):
        from MAIN_WINDOW import MainWindow
        from PySide6.QtWidgets import QDialog
        window = MainWindow()
        try:
            window.control_panel.load_simulated_demo()
            configs, _ = window.control_panel.collect_current_configuration()
            creator = fixture_creator()
            campaign = DoeCampaignService(creator).compile(fixture_assignments(), configs)
            window.create_experiments = creator
            window.settings_page.set_doe_available(True)
            dialog = Mock(campaign=campaign)
            dialog.exec.return_value = QDialog.DialogCode.Accepted
            dialog.assignments.return_value = fixture_assignments()
            with patch("MAIN_WINDOW.DoeAssignmentDialog", return_value=dialog), patch.object(window.control_panel, "start_campaign") as start:
                window.assign_doe_devices()
                start.assert_not_called()
                self.assertTrue(window.settings_page.start_independent_doe_button.isEnabled())
                window.settings_page.start_independent_doe_button.click()
                start.assert_called_once_with(campaign)
                self.assertFalse(window.execution.timer.isActive())
                self.assertFalse(window.execution.campaign_mode)
                with patch.object(window, "_start_loop_campaign") as loop:
                    window._choose_doe_campaign(True)
                    loop.assert_called_once_with(campaign)
                start.reset_mock()
                window.control_panel.device_widgets["TEM_MAIN"].config["simulated"] = False
                window.start_independent_doe()
                start.assert_not_called()
                self.assertFalse(window.settings_page.start_independent_doe_button.isEnabled())
        finally:
            window.close()
            APP.processEvents()

    def test_control_panel_folder_round_trip_and_legacy_default(self):
        from MAIN_WINDOW import MainWindow
        from app.paths import DATA_DIR
        window = MainWindow()
        panel = window.control_panel
        try:
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as parent:
                filename = str(Path(parent) / "control.json")
                folder = str(Path(parent) / "manual")
                panel.control_output_path.setText(folder)
                with patch("control_panel.control_panel.QFileDialog.getSaveFileName", return_value=(filename, "")), patch("control_panel.control_panel.QMessageBox.information"):
                    panel.save_configuration()
                self.assertEqual(json.loads(Path(filename).read_text())["recording_folder"], folder)
                panel.control_output_path.setText("different")
                with patch("control_panel.control_panel.QFileDialog.getOpenFileName", return_value=(filename, "")), patch("control_panel.control_panel.QMessageBox.information"):
                    panel.load_configuration()
                self.assertEqual(panel.control_output_path.text(), folder)
                configs, _ = panel.collect_current_configuration()
                panel.apply_loaded_configuration(configs, {}, [])
                self.assertEqual(panel.control_output_path.text(), str(DATA_DIR / "control_panel_runs"))
        finally:
            window.close()
            APP.processEvents()

    def test_preview_without_density_and_changes_invalidate_approval(self):
        configs = fixture_configs()
        hardware = create_hardware(configs, HARDWARE_TYPES)
        for device in hardware.values():
            device.connect()
        dialog = DoeAssignmentDialog(fixture_creator(), configs, hardware)
        try:
            dialog.preview()
            self.assertIsNotNone(dialog.campaign)
            self.assertEqual(dialog.table.columnCount(), 3)
            self.assertTrue(dialog.start_button.isEnabled())
            self.assertIn("SCL_AS", dialog.campaign.monitor_device_ids)
            dialog.rows["AS"][0].setCurrentIndex(0)
            self.assertIsNone(dialog.campaign)
            self.assertFalse(dialog.start_button.isEnabled())
        finally:
            dialog.close()

    def test_assignments_save_and_load(self):
        configs = fixture_configs()
        dialog = DoeAssignmentDialog(fixture_creator(), configs, create_hardware(configs, HARDWARE_TYPES))
        try:
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as parent:
                filename = str(Path(parent) / "assignments.json")
                with patch("gui.doe_assignment_dialog.QFileDialog.getSaveFileName", return_value=(filename, "")):
                    dialog.save_assignments()
                dialog.rows["AS"][0].setCurrentIndex(0)
                with patch("gui.doe_assignment_dialog.QFileDialog.getOpenFileName", return_value=(filename, "")):
                    dialog.load_assignments()
                self.assertEqual(dialog.rows["AS"][0].currentData(), "PMP_AS")
                self.assertIsNone(next(a for a in dialog.assignments() if a.role == "AS").density_g_ml)
        finally:
            dialog.close()

    def test_campaign_launch_controls_only_assigned_devices(self):
        from MAIN_WINDOW import MainWindow
        window = MainWindow()
        panel = window.control_panel
        try:
            panel.load_simulated_demo()
            configs, _ = panel.collect_current_configuration()
            # All unrelated devices deliberately remain disconnected.
            config = {"TEM_MAIN": configs["TEM_MAIN"]}
            creator = fixture_creator()
            for frame in creator.exp_storage.data.values():
                for role in frame.columns:
                    if role not in {"t", "T"}:
                        frame[role] = 0
            campaign = DoeCampaignService(creator).compile([ProfileAssignment("T", "TEM_MAIN")], config)
            panel.hardware["TEM_MAIN"].connect()
            panel.device_widgets["TEM_MAIN"]._update_connection_state()
            panel.simulation_speed.setValue(100)
            panel.start_campaign(campaign)
            dashboard = window.experiment_page.dashboard
            self.assertEqual(dashboard.recording_settings["output_folder"], str(Path.cwd() / "data" / "runs"))
            self.assertNotEqual(dashboard.recording_settings["output_folder"], panel.control_output_path.text())
            self.assertEqual(set(dashboard.devices), {"TEM_MAIN"})
            self.wait_for(lambda: hasattr(dashboard, "last_result") and dashboard.thread is None)
            self.assertEqual(dashboard.last_result["status"], "completed")
            self.assertEqual(dashboard.last_result["campaign_id"], campaign.campaign_id)
            self.assertFalse(window.execution.campaign_mode)
            self.assertFalse(window.execution.timer.isActive())
            self.assertFalse(panel.hardware["STR_MAIN"].is_connected)
            window.end_experiment_session()
        finally:
            dashboard = window.experiment_page.dashboard
            if dashboard is not None:
                self.wait_for(lambda: dashboard.thread is not None or not dashboard.is_execution_active())
                dashboard.stop_orchestrator()
                self.wait_for(lambda: dashboard.thread is None)
                window.end_experiment_session()
            window.close()
            APP.processEvents()

    def test_generation_needs_no_folders_and_folder_start_uses_actual_factorial_count(self):
        from MAIN_WINDOW import MainWindow
        window = MainWindow()
        try:
            window.settings_page.exp_number_input.setText("2")
            window._assigned_doe_campaign = object()
            window.settings_page.set_assignment_available(True)
            window.settings_page.doe_type_combo.setCurrentText("Full Factorial DoE generation")
            with patch.object(window.settings_page, "render_doe_plots"):
                window.settings_page.generate_button.click()
            self.assertIsNone(window.CONTROLLER)
            self.assertIsNotNone(window.create_experiments)
            self.assertIsNone(window._assigned_doe_campaign)
            self.assertFalse(window.settings_page.start_independent_doe_button.isEnabled())
            self.assertTrue(window.settings_page.assign_doe_button.isEnabled())
            self.assertTrue(window.settings_page.export_doe_button.isEnabled())
            actual = window.create_experiments.actually_sampled_experiments
            self.assertGreater(actual, 2)
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as parent:
                omni, model = Path(parent) / "omni", Path(parent) / "model"
                omni.mkdir()
                model.mkdir()
                window.settings_page.exp_path_input.setText(str(omni))
                window.settings_page.MODEL_path_input.setText(str(model))
                self.assertTrue(window.define_command())
                with patch.object(window, "_choose_doe_campaign") as start:
                    window.dashboard_page.start_button.click()
                    window.quick_start_btn.click()
                    self.assertEqual(start.call_count, 2)
                    start.assert_called_with(True)
        finally:
            window.close()
            APP.processEvents()

