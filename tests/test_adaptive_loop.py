"""Exercise dashboard-driven modeling with the recorded validation DoE."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import numpy as np
import pandas as pd

from app.presets import read_preset
from experiment.adaptive_loop import AdaptiveLoop, surviving_models
from experiment.doe_campaign import DoeCampaignService
from experiment.doe_snapshot import restore_doe
from imed_python.demo_campaign import model_inputs
from MAIN_WINDOW import MainWindow


APP = QApplication.instance() or QApplication([])
PRESET = Path(__file__).resolve().parents[1] / "examples/configurations/imed_validation_doe.json"


class AdaptiveLoopTests(unittest.TestCase):
    def setup_dashboard(self, root):
        window = MainWindow()
        self.addCleanup(self.close_window, window)
        panel = window.control_panel
        data, parameters, assignments = read_preset(PRESET)
        self.assertTrue(panel.apply_loaded_configuration(data["control_panel"]["devices"], {}, []))
        for key, device in panel.hardware.items():
            device.connect()
            panel.device_widgets[key]._update_connection_state()
        panel.simulation_speed.setValue(100)
        service = window.experiment_service
        service.exp_par = parameters
        service._build_doe_dependencies()
        window.create_experiments = restore_doe(service, data["generated_doe"])
        doe = DoeCampaignService(window.create_experiments)
        configs, _ = panel.collect_current_configuration()
        campaign = doe.compile(assignments, configs, data["monitor_device_ids"])
        window._doe_assignments = assignments
        window._apply_doe_campaign(campaign, configs)
        experiments = root / "experiments"
        models = root / "models"
        experiments.mkdir()
        models.mkdir()
        window.settings_page.exp_path_input.setText(str(experiments))
        window.settings_page.MODEL_path_input.setText(str(models))
        window.settings_page.loop_backend.setCurrentIndex(window.settings_page.loop_backend.findData("bo"))
        window.bo_page.target_fields["d50"][0].setChecked(True)
        logs = []
        window.log_message = lambda message, level="INFO": logs.append((level, message))
        return window, experiments, models, logs

    def close_window(self, window):
        window.stop_command()
        self.wait_for(lambda: window._loop_model_job is None and not window.control_panel.hardware_manager.is_busy)
        window.end_experiment_session()
        window.close()
        APP.processEvents()

    @staticmethod
    def wait_for(condition, timeout=35):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            APP.processEvents()
            if condition():
                return True
            sleep(.01)
        return False

    def test_dashboard_runs_the_bo_recommendation_once(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window, experiments, models, logs = self.setup_dashboard(Path(directory))
            try:
                window.bo_page.restore(json.loads(PRESET.with_name("bo_validation_loop.json").read_text()))
                window.dashboard_page.start_button.click()
                self.assertIsNotNone(window._adaptive_loop, logs)
                self.assertTrue(self.wait_for(lambda: window._adaptive_loop is None), logs)
                self.assertFalse(any(level == "ERROR" for level, _ in logs), logs)
                self.assertEqual(window.execution.model_run_count, 1)
                recommendations = list(models.glob("*/recommendation.json"))
                self.assertEqual(len(recommendations), 1)
                recipe = json.loads(recommendations[0].read_text())
                followup = recipe["campaign"]["campaign_id"]
                summaries = list(experiments.glob("*/summary.json"))
                self.assertEqual(len(summaries), 2)
                output = next(path for path in summaries if f"campaign_{followup}_" in path.parent.name)
                summary = json.loads(output.read_text())
                self.assertEqual(summary["status"], "completed")
                self.assertTrue(summary["recording_complete"])
                self.assertEqual(len(summary["runs"]), 1)
                self.assertEqual(summary["runs"][0]["experiment_id"], recipe["campaign"]["experiments"][0]["experiment_id"])
            finally:
                self.doCleanups()

    def test_manual_batch_can_drive_bo_with_explicit_actuator_mapping(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window, experiments, models, logs = self.setup_dashboard(Path(directory))
            try:
                window.control_panel.experiments = [pd.DataFrame({"time_s": [0., 12., 30.], "T1": [t] * 3})
                                                    for t in (14., 19., 24.)]
                _, _, process, solubility, _ = model_inputs(False)
                window.imed_page.panel.restore_setup_settings({"Solubility": solubility})
                setup = window.settings_page.loop_setup
                setup["max_followups"] = 2
                setup["manual_assignments"] = {"T": "T1"}
                setup["process"] = process
                setup["recording"].update(no_feed=True, initial_concentration=.1)
                setup["channels"] = {"Concentration": ""}
                bo = json.loads(PRESET.with_name("bo_validation_loop.json").read_text())
                bo["targets"]["d50"]["enabled"] = False
                bo["targets"]["yield"]["enabled"] = True
                bo["physical_profiles"] = {}
                window.bo_page.restore(bo)
                with patch.object(window, "_choose_loop_source", return_value="manual"):
                    window.start_command()
                self.assertIsNotNone(window._adaptive_loop, logs)
                self.assertTrue(self.wait_for(lambda: window._adaptive_loop is None), logs)
                self.assertFalse(any(level == "ERROR" for level, _ in logs), logs)
                results = [json.loads(path.read_text()) for path in sorted(models.glob("*/recommendation.json"))]
                self.assertEqual([result["observations"] for result in results], [3, 4])
                for result in results:
                    followup = next(experiments.glob(f"campaign_{result['campaign']['campaign_id']}_*/summary.json"))
                    self.assertTrue(json.loads(followup.read_text())["recording_complete"])
            finally:
                self.doCleanups()

    def test_dashboard_runs_imed_model_discrimination_and_d_optimal_design(self):
        for count, method in ((2, "Model discrimination"), (1, "D-optimal design")):
            with self.subTest(method=method), TemporaryDirectory(dir=Path.cwd()) as directory:
                window, experiments, models, logs = self.setup_dashboard(Path(directory))
                try:
                    settings = window.settings_page
                    settings.loop_backend.setCurrentIndex(settings.loop_backend.findData("imed"))
                    parameters, _, process, solubility, mesh = model_inputs(True)
                    paraset = np.column_stack((np.zeros(len(parameters)), parameters, parameters, parameters))
                    for index in (0, 12):
                        value = parameters[index]
                        paraset[index] = (1, value - .04, value - .25, value + .25)
                    case = {"ModelParameter": [{"Name": f"Candidate {i + 1}", "paraset": paraset.copy()}
                                               for i in range(count)],
                            "Solubility": solubility, "pbmsetting": mesh,
                            "objweightfactor": {"W1": 0., "W2": 1., "W3": 1.},
                            "pyfitsettings": {"global_iterations": 0, "population_size": 5, "local_evaluations": 40}}
                    if count > 1:
                        window.imed_page.panel.restore_setup_settings(case)
                    seed = float(process["SeedMass"])
                    settings.loop_setup = json.loads(PRESET.with_name("imed_validation_loop.json").read_text())
                    self.assertAlmostEqual(settings.loop_setup["imed"]["conditions"]["seed_bounds"][0], seed)
                    window.start_command()
                    self.assertIsNotNone(window._adaptive_loop, logs)
                    self.assertTrue(self.wait_for(lambda: window._adaptive_loop is None, timeout=120), logs)
                    self.assertFalse(any(level == "ERROR" for level, _ in logs), logs)
                    recommendation, = models.glob("*/recommendation.json")
                    result = json.loads(recommendation.read_text())
                    self.assertEqual(result["method"], method)
                    self.assertEqual(len(result["surviving_models"]), count)
                    self.assertTrue(recommendation.with_name("fit.imed.json").exists())
                    followup = next(experiments.glob(f"campaign_{result['campaign']['campaign_id']}_*/summary.json"))
                    self.assertTrue(json.loads(followup.read_text())["recording_complete"])
                finally:
                    self.doCleanups()

    def test_stop_while_modeling_prevents_a_followup(self):
        started = Event()

        def wait_for_stop(session, manifest, cancelled, report):
            started.set()
            deadline = monotonic() + 15
            while not cancelled() and monotonic() < deadline:
                sleep(.01)
            raise InterruptedError("Stopped")

        with TemporaryDirectory(dir=Path.cwd()) as directory, patch.object(AdaptiveLoop, "recommend", wait_for_stop):
            window, experiments, models, logs = self.setup_dashboard(Path(directory))
            try:
                window.start_command()
                self.assertTrue(self.wait_for(started.is_set), logs)
                window.stop_command()
                self.assertTrue(self.wait_for(lambda: window._loop_model_job is None), logs)
                self.assertEqual(len(list(experiments.glob("*/definition.json"))), 1)
                self.assertFalse(list(models.glob("*/recommendation.json")))
            finally:
                self.doCleanups()

    def test_stop_before_scheduled_start_releases_the_reservation(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window, experiments, _, logs = self.setup_dashboard(Path(directory))
            try:
                window.start_command()
                dashboard = window.experiment_page.dashboard
                self.assertTrue(dashboard.is_execution_active(), logs)
                window.stop_command()
                dashboard.start_scheduled_campaign()
                self.assertFalse(dashboard.is_execution_active())
                self.assertFalse(list(experiments.glob("*/definition.json")))
            finally:
                self.doCleanups()

    def test_preflight_leaves_hardware_idle_when_a_folder_and_monitor_are_missing(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window, experiments, _, logs = self.setup_dashboard(Path(directory))
            try:
                window.settings_page.MODEL_path_input.setText(str(Path(directory) / "missing"))
                window.control_panel.hardware["D501"].disconnect()
                window.control_panel.device_widgets["D501"]._update_connection_state()
                window.start_command()
                messages = "\n".join(message for _, message in logs)
                self.assertIn("Choose an existing MODEL folder", messages)
                self.assertIn("Connect assigned devices and monitors: D501", messages)
                self.assertFalse(window.control_panel.hardware_manager.is_busy)
                self.assertIsNone(window.experiment_page.dashboard)
                self.assertFalse(list(experiments.iterdir()))
            finally:
                self.doCleanups()

    def test_model_survival_uses_total_aic_units(self):
        fits = {0: SimpleNamespace(aic=3.), 1: SimpleNamespace(aic=3.01), 2: SimpleNamespace(aic=3.07)}
        self.assertEqual(surviving_models(fits, 100, 2.)[0], (0, 1))
        self.assertEqual(surviving_models(fits, 500, 2.)[0], (0,))
        fits[2].aic = float("nan")
        with self.assertRaisesRegex(ValueError, "finite IMED AIC"):
            surviving_models(fits, 100, 2.)
