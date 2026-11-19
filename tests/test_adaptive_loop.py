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

from PySide6.QtWidgets import QApplication, QDialog, QListWidget, QTabWidget, QTableWidget
import numpy as np
import pandas as pd

from app.presets import read_preset
from experiment.adaptive_loop import AdaptiveLoop, optimize_recipe
from experiment.doe_campaign import DoeCampaignService
from experiment.doe_snapshot import restore_doe
from imed_python.demo_campaign import model_inputs
from imed_python.model_screening import screen_models
from imed_python.native_case import read_native_case
from imed_python.session import history, latest_pe_step, loop_stages
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

    def check_five_run_preset(self, backend, source="doe", *, load_from_control_panel=False):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            window = MainWindow()
            logs = []
            stages = []
            window.log_message = lambda message, level="INFO": logs.append((level, message))
            show_stage = window._loop_views.show_stage

            def observe(stage, value, job):
                show_stage(stage, value, job)
                stages.append(stage)

            window._loop_views.show_stage = observe
            try:
                suffix = "setup" if source == "doe" else "control_panel"
                preset = PRESET.with_name(f"{backend}_five_run_{suffix}.json")
                with patch("app.paths.DATA_DIR", root), \
                     patch("MAIN_WINDOW.QFileDialog.getOpenFileName", return_value=(str(preset), "")), \
                     patch("MAIN_WINDOW.QMessageBox.warning") as warning, \
                     patch("MAIN_WINDOW.QMessageBox.critical") as critical:
                    if load_from_control_panel:
                        window.control_panel.load_config_button.click()
                    else:
                        window.load_setup_preset()
                warning.assert_not_called()
                critical.assert_not_called()
                self.assertIs(window.stack.currentWidget(), window.dashboard_page)
                self.assertTrue(all(device.simulated and device.is_connected for device in window.control_panel.hardware.values()))
                if source == "doe":
                    self.assertEqual(len(window._assigned_doe_campaign.experiments), 5)
                else:
                    self.assertEqual(len(window.control_panel.experiments), 5)
                    self.assertIsNone(window.create_experiments)
                if backend == "imed":
                    candidates = window.imed_page.panel.case["ModelParameter"]
                    self.assertEqual(len(candidates), 2)
                    self.assertFalse(np.array_equal(candidates[0]["paraset"], candidates[1]["paraset"]))
                output = root / "runs" / "simulated_loops" / backend
                self.assertFalse(list((output / "experiments").glob("campaign_*")))
                window.dashboard_page.start_button.click()
                self.assertIsNotNone(window._adaptive_loop, logs)
                self.assertTrue(self.wait_for(lambda: window._adaptive_loop is None, timeout=300), logs)
                self.assertFalse(any(level == "ERROR" for level, _ in logs), logs)
                recommendation, = (output / "models").glob("*/recommendation.json")
                recipe = json.loads(recommendation.read_text())
                self.assertIn(recipe["method"], {"BO"} if backend == "bo" else {"D-optimal design", "Model discrimination"})
                self.assertEqual(recipe["observations"], 5)
                summaries = [json.loads(path.read_text()) for path in (output / "experiments").glob("*/summary.json")]
                self.assertEqual(sorted(len(summary["runs"]) for summary in summaries), [1, 5])
                self.assertTrue(all(summary["status"] == "completed" and summary["recording_complete"]
                                    for summary in summaries))
                self.assertEqual(len(list(recommendation.parent.glob("inputs/*/execution.csv"))), 5)
                self.assertTrue(recommendation.with_name("device_targets.csv").is_file())
                self.assertEqual(len(list((output / "models").glob("*/inputs/*/execution.csv"))), 6)
                if backend == "bo":
                    self.assertEqual(stages, ["bo_inputs", "bo_result", "bo_stage", "bo_inputs", "bo_stage"])
                    self.assertEqual(len(window.bo_page.state["runs"]), 6)
                    self.assertIsNotNone(window.bo_page.proposal)
                    self.assertTrue((recommendation.parent / "bo" / "predictions.csv").is_file())
                    first, last = window.bo_page.state["loop_stages"]
                    self.assertEqual([first["name"], last["name"]], ["BO 1", "BO 2"])
                    self.assertEqual([first["observations"], last["observations"]], [5, 6])
                    first_case = json.loads(Path(first["path"]).read_text())
                    last_case = json.loads(Path(last["path"]).read_text())
                    self.assertTrue(first_case["loop_proposal"]["profiles"])
                    self.assertTrue(first_case["loop_proposal"]["predictions"])
                    self.assertNotIn("loop_proposal", last_case)
                    self.assertEqual(last_case["runs"][-1]["recipe_stage"], "BO 1")

                    def check_bo_stages(dialog):
                        choices = dialog.findChild(QListWidget)
                        self.assertEqual(choices.count(), 2)
                        tabs = dialog.findChild(QTabWidget)
                        measurements, predictions = tabs.widget(0), tabs.widget(1)
                        self.assertEqual(measurements.rowCount(), 6)
                        choices.setCurrentRow(0)
                        self.assertEqual(measurements.rowCount(), 5)
                        self.assertGreater(predictions.rowCount(), 0)
                        return QDialog.DialogCode.Rejected

                    with patch.object(QDialog, "exec", check_bo_stages):
                        window.bo_page.stages_button.click()
                else:
                    self.assertEqual([stage for stage in stages if stage != "imed_generation"],
                                     ["imed_inputs", "imed_pe", "imed_fits", "imed_design", "imed_result",
                                      "imed_inputs", "imed_pe", "imed_fits"])
                    panel = window.imed_page.panel
                    self.assertEqual(len(panel.case["ExpData"]), 6)
                    survivors = recipe["surviving_models"]
                    self.assertIn(f"Estimation runs: {2 + len(survivors)}", panel.status.toPlainText())
                    self.assertIn("Experiment 6 · " + recipe["method"], panel.followup_summary.text())
                    readout = window._loop_views.readout
                    self.assertTrue(readout.isVisible())
                    self.assertEqual(len(readout.case["ExpData"]), 6)
                    self.assertTrue(readout.details_button.isEnabled())
                    stage_names = ["PE 1", recipe["stage_name"], "PE 2"]
                    stages = loop_stages(panel.case)
                    self.assertEqual([stage["name"] for stage in stages], stage_names)
                    self.assertEqual([row["model"] for row in stages[-1]["candidates"]], survivors)
                    for stage in stages:
                        path = Path(stage["case_file"])
                        snapshot = read_native_case(path)
                        self.assertEqual(len(snapshot["ExpData"]), stage["experiments"])
                        self.assertEqual(loop_stages(snapshot)[-1]["name"], stage["name"])

                    def check_stages(dialog, layout):
                        stage_list = dialog.findChild(QListWidget)
                        self.assertEqual(stage_list.count(), 3)
                        self.assertEqual(stage_list.currentRow(), 2)
                        self.assertEqual(dialog.findChild(QTableWidget).rowCount(), len(survivors))
                        stage_list.setCurrentRow(0)
                        self.assertEqual(dialog.findChild(QTableWidget).rowCount(), 2)

                    with patch.object(panel, "show_dialog", side_effect=check_stages):
                        panel.show_loop_stages()
                    self.assertEqual(len(pd.read_csv(recommendation.parent / "imed" / "pe_models.csv")), 2)
                messages = "\n".join(message for _, message in logs)
                self.assertIn("Saved the", messages)
                self.assertIn("Launched automatic follow-up", messages)
                self.assertIn("Control Panel profiles" if source != "doe" else "Settings DoE", messages)
                if backend == "bo":
                    self.check_execute_bo_proposal(window, output, logs)
            finally:
                self.close_window(window)

    def check_execute_bo_proposal(self, window, output, logs):
        previous = window.experiment_page.dashboard
        self.assertTrue(self.wait_for(lambda: not previous.is_execution_active()), logs)
        controller = window.control_panel.device_widgets["T1"]
        controller.device.disconnect()
        controller._update_connection_state()
        window.bo_page.execute_button.click()
        self.assertIs(window.experiment_page.dashboard, previous)
        self.assertIn("Connect assigned devices and monitors: T1", window.bo_page.status.text())
        self.assertTrue(controller.connect_device())
        window.bo_page.execute_button.click()
        current = window.experiment_page.dashboard
        self.assertIsNot(current, previous)
        self.assertIsNotNone(current)
        summaries = output / "experiments"
        self.assertTrue(self.wait_for(lambda: len(list(summaries.glob("*/summary.json"))) == 3
                                     and not current.is_execution_active()), logs)
        self.assertTrue(all(json.loads(path.read_text())["status"] == "completed"
                            for path in summaries.glob("*/summary.json")))
        self.assertEqual(len(list((output / "models").glob("*/recommendation.json"))), 1)
        self.assertTrue(self.wait_for(lambda: len(window.bo_page.state["runs"]) == 7), logs)
        self.assertIn(6, window.bo_page.state["selected"])
        self.assertIsNone(window.bo_page.proposal)
        self.assertTrue(Path(window.bo_page.state["runs"][-1]["recording"]).is_file())
        imported, = (output / "models").glob("_recipe_bo_*/recipe_inputs.bo.json")
        self.assertEqual(len(json.loads(imported.read_text())["runs"]), 7)

    def test_load_and_start_bo_five_run_preset(self):
        self.check_five_run_preset("bo")

    def test_load_and_start_imed_five_run_preset(self):
        self.check_five_run_preset("imed")

    def test_load_and_start_bo_control_panel_preset(self):
        self.check_five_run_preset("bo", "control-panel", load_from_control_panel=True)

    def test_load_and_start_imed_control_panel_preset(self):
        self.check_five_run_preset("imed", "control-panel")

    def test_execute_previous_imed_design_as_a_single_run(self):
        from imed_python.design_dialog import DesignDialog
        from imed_python.session import record_step

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window = MainWindow()
            try:
                preset = PRESET.with_name("imed_five_run_control_panel.json")
                with patch("app.paths.DATA_DIR", Path(directory)), patch("MAIN_WINDOW.QMessageBox.warning") as warning:
                    window.load_setup_preset(str(preset))
                warning.assert_not_called()
                panel = window.imed_page.panel
                _, experiment, process, _, _ = model_inputs(True)
                experiment = {**experiment, "TemperatureData": np.array([[0., 40.], [18., 18.]]),
                              "FlowRateData": np.array([[0., 40.], [0., 0.]])}
                panel.case.update(ExpData=[experiment], ProcInf=[process])
                model = panel.case["ModelParameter"][0]
                model.update(para=model["paraset"][model["paraset"][:, 0] == 1, 1], covmat=np.eye(2))
                record_step(panel.case, "d_optimal_design", {"score": 1., "experiment": experiment,
                            "process": process, "request": {}}, context="earlier inputs")

                def execute(dialog):
                    self.assertIn("Previous design", dialog.status.text())
                    self.assertTrue(dialog.execute_button.isEnabled())
                    dialog.execute_button.click()
                    return dialog.result()

                with patch.object(DesignDialog, "exec", execute):
                    panel.show_design(True)
                dashboard = window.experiment_page.dashboard
                self.assertIsNotNone(dashboard)
                output = Path(directory) / "runs" / "simulated_loops" / "imed" / "experiments"
                self.assertTrue(self.wait_for(lambda: bool(list(output.glob("*/summary.json")))
                                             and not dashboard.is_execution_active()))
                summary, = output.glob("*/summary.json")
                result = json.loads(summary.read_text())
                self.assertEqual(result["status"], "completed")
                self.assertEqual(len(result["runs"]), 1)
                self.assertFalse(window.execution.timer.isActive())
                self.assertTrue(self.wait_for(lambda: len(panel.case["ExpData"]) == 2))
                added = panel.case["ExpData"][-1]
                self.assertEqual(added["RecipeSource"], "D-optimal design")
                self.assertTrue(Path(added["RecordingFile"]).is_file())
                self.assertEqual(panel.case["ProcInf"][-1]["No"], 2)
                models = output.parent / "models"
                imported, = models.glob("_recipe_imed_*/recipe_inputs.imed.json")
                self.assertEqual(len(read_native_case(imported)["ExpData"]), 2)
            finally:
                self.close_window(window)

    def test_automatic_preset_preparation_rejects_real_device_configs(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            preset = Path(directory) / "mixed.json"
            mixed = json.loads(PRESET.with_name("imed_five_run_setup.json").read_text())
            mixed["control_panel"]["devices"]["T1"]["simulated"] = False
            preset.write_text(json.dumps(mixed), encoding="utf-8")
            window = MainWindow()
            try:
                with patch("MAIN_WINDOW.QFileDialog.getOpenFileName", return_value=(str(preset), "")), \
                     patch("MAIN_WINDOW.QMessageBox.warning") as warning, \
                     patch.object(window.control_panel, "apply_loaded_configuration") as apply:
                    window.load_setup_preset()
                apply.assert_not_called()
                self.assertIn("simulated experiment source", warning.call_args.args[2])
            finally:
                self.close_window(window)

    def test_manual_batch_can_drive_bo_with_explicit_actuator_mapping(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window, experiments, models, logs = self.setup_dashboard(Path(directory))
            try:
                window.control_panel.experiments = [pd.DataFrame({"time_s": [0., 12., 30.], "T1": [t] * 3})
                                                    for t in (14., 19., 24.)]
                _, _, process, solubility, _ = model_inputs(False)
                window.imed_page.panel.restore_setup_settings({"Solubility": solubility})
                setup = window.settings_page.loop_setup
                setup["manual_assignments"] = {"T": "T1"}
                setup["process"] = process
                setup["recording"].update(no_feed=True, initial_concentration=.1)
                setup["channels"] = {"Concentration": ""}
                bo = json.loads(PRESET.with_name("bo_validation_loop.json").read_text())
                bo["targets"]["d50"]["enabled"] = False
                bo["targets"]["yield"]["enabled"] = True
                bo["physical_profiles"] = {}
                window.bo_page.restore(bo)

                def set_followups(dialog):
                    dialog.max_followups.setValue(2)
                    dialog.accept()
                    return dialog.result()

                dialog = window.settings_page.edit_loop_setup()
                set_followups(dialog)
                self.assertEqual(window.settings_page.loop_setup["max_followups"], 2)
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

    def test_loop_setup_shows_applied_and_draft_values_for_each_source(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window, _, _, _ = self.setup_dashboard(Path(directory))
            try:
                window.control_panel.experiments = [pd.DataFrame({"time_s": [0., 10.], "T1": [18., 18.]})]
                window.imed_page.panel.restore_setup_settings({"ProcInf": [{"SolvMass": 75.}]})
                settings = window.settings_page
                settings.loop_setup["manual_assignments"] = {"T": "T1"}
                settings.loop_setup["channels"]["Temperature"] = "T1.temp_internal"

                def inspect_preview(dialog):
                    def rows(section):
                        tree = dialog.used_values
                        group = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                                     if tree.topLevelItem(i).text(0) == section)
                        return {group.child(i).text(0): (group.child(i).text(1), group.child(i).text(2))
                                for i in range(group.childCount())}

                    self.assertIs(dialog.tabs.currentWidget(), dialog.used_values_tab)
                    self.assertEqual(dialog.preview_source.currentData(), "doe")
                    self.assertEqual(rows("Device assignments")["Reactor temperature"][1], "Applied DoE assignments")
                    self.assertEqual(rows("Recording")["Temperature"], ("T1.temp_external", "PBM simulation"))
                    dialog.preview_source.setCurrentIndex(dialog.preview_source.findData("manual"))
                    self.assertEqual(rows("Device assignments")["Reactor temperature"][1], "Control Panel assignments")
                    self.assertEqual(rows("Recording")["Temperature"], ("T1.temp_internal", "Recording override"))
                    self.assertIn("18 °C · Shared profile", rows("Starting profiles")["T1"][0])
                    self.assertIn(("75", "First IMED process record"), rows("Starting process inputs").values())
                    edit_tab = dialog.tabs.indexOf(dialog.used_values_tab) + 1
                    dialog.tabs.setCurrentIndex(edit_tab)
                    dialog.process["SolvMass"].setText("80")
                    dialog.tabs.setCurrentWidget(dialog.used_values_tab)
                    self.assertIn(("80", "Process inputs override"), rows("Starting process inputs").values())
                    return dialog.result()

                dialog = settings.edit_loop_setup()
                inspect_preview(dialog)
                dialog.reject()
                self.assertNotIn("SolvMass", settings.loop_setup["process"])
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
                    def check_fit_saved_before_design(*args, **kwargs):
                        path, = models.glob("*/fit.imed.json")
                        saved = read_native_case(path)
                        self.assertEqual(len(saved["ExpData"]), 3)
                        self.assertTrue(all(latest_pe_step(saved, index) for index in range(count)))
                        return optimize_recipe(*args, **kwargs)

                    with patch("experiment.adaptive_loop.optimize_recipe", side_effect=check_fit_saved_before_design):
                        self.assertTrue(self.wait_for(lambda: window._adaptive_loop is None, timeout=180), logs)
                    self.assertFalse(any(level == "ERROR" for level, _ in logs), logs)
                    recommendation, = models.glob("*/recommendation.json")
                    result = json.loads(recommendation.read_text())
                    self.assertEqual(result["method"], method)
                    self.assertEqual(len(result["surviving_models"]), count)
                    fit_path = recommendation.with_name("fit.imed.json")
                    saved_fit = read_native_case(fit_path)
                    self.assertEqual(len(saved_fit["ExpData"]), 3)
                    final_path, = (path for path in models.glob("*/fit.imed.json") if path != fit_path)
                    final = read_native_case(final_path)
                    self.assertEqual(len(final["ExpData"]), 4)
                    for index in range(count):
                        self.assertIsNotNone(latest_pe_step(saved_fit, index))
                        self.assertIsNotNone(latest_pe_step(final, index))
                    self.assertEqual(final["ExpData"][-1]["RecipeSource"], method)
                    self.assertEqual(final["ExpData"][-1]["RecipeStage"], "MD 1" if count > 1 else "DO 1")
                    self.assertEqual(final["ExpData"][-1]["RecipeModelFolder"], str(recommendation.parent))
                    diagnostics = pd.read_csv(fit_path.parent / "imed" / "pe_models.csv")
                    self.assertTrue(diagnostics["f_pass"].all() and diagnostics["t_pass"].all())
                    self.assertEqual(set(diagnostics["aic_rank"]), set(range(1, count + 1)))
                    inputs = pd.read_csv(final_path.parent / "imed" / "process_inputs.csv")
                    self.assertEqual(inputs.iloc[-1]["recipe_source"], method)
                    action = "model_discrimination" if count > 1 else "d_optimal_design"
                    designs = [step for step in history(final) if step["action"] in ("model_discrimination", "d_optimal_design")]
                    self.assertEqual([step["action"] for step in designs], [action])
                    self.assertEqual(designs[0]["result"]["reason"], result["design_reason"])
                    if count == 1:
                        self.assertTrue(any("skipped model discrimination" in message for _, message in logs))
                    followup = next(experiments.glob(f"campaign_{result['campaign']['campaign_id']}_*/summary.json"))
                    self.assertTrue(json.loads(followup.read_text())["recording_complete"])
                finally:
                    self.doCleanups()

    def test_stop_while_modeling_prevents_a_followup(self):
        started = Event()

        def wait_for_stop(session, manifest, cancelled, report, **kwargs):
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
        fits = {index: SimpleNamespace(aic=aic, ssr=1., converged=True,
                                      parameters=np.array([3.]), covariance=np.eye(1))
                for index, aic in enumerate((3., 3.01, 3.07))}
        for samples, expected in ((100, (0, 1)), (500, (0,))):
            results = screen_models(fits, sample_count=samples, delta_aic=2.)
            self.assertEqual(tuple(index for index, result in results.items() if not result["reasons"]), expected)
        fits[2].aic = float("nan")
        self.assertIn("AIC unavailable", screen_models(fits, sample_count=100, delta_aic=2.)[2]["reasons"])

    def test_matlab_f_and_t_checks_identify_rejected_models(self):
        fits = {index: SimpleNamespace(aic=aic, ssr=ssr, converged=True,
                                       parameters=np.array([parameter]), covariance=np.eye(1))
                for index, (aic, ssr, parameter) in enumerate(((1., 1., -3.), (.8, 1.1, .5), (1.01, 4., 3.)))}
        results = screen_models(fits, sample_count=100, delta_aic=2.)
        self.assertAlmostEqual(results[0]["f_limit"], 2.6049093010837265)
        self.assertAlmostEqual(results[0]["t_limit"], 1.959963984540054)
        self.assertEqual(results[0]["reasons"], [])
        self.assertEqual(results[0]["rank"], 2)
        self.assertEqual(results[0]["delta_aic"], 0.)
        self.assertEqual(results[1]["reasons"], ["parameter t-test failed"])
        self.assertEqual(results[2]["reasons"], ["F-test failed"])
        fits[0].covariance[0, 0] = np.nan
        self.assertIn("parameter t-tests unavailable", screen_models(fits)[0]["reasons"])
