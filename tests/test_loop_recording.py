import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from threading import Event
from unittest.mock import patch
import tempfile
import unittest
import json
import pandas as pd
from PySide6.QtWidgets import QApplication
from experiment.recording import RunRecorder, validate_settings
from experiment.models import Campaign, ExperimentDefinition
from experiment.execution_coordinator import ExperimentCoordinator
from app.controller import Controller, Stage

APP = QApplication.instance() or QApplication([])


class LoopRecordingTests(unittest.TestCase):
    def test_skip_idles_devices_and_sends_only_completed_run_to_model(self):
        from experiment_dashboard.experiment_dashboard import ExperimentWorker
        from hardware_devices.simulated import SimulatedDevice

        configs = {"pump": {"type": "masterflex"}}
        first = pd.DataFrame({"time_s": [0, 300], "pump": [10, 10]})
        second = pd.DataFrame({"time_s": [0], "pump": [20]})
        pump = SimulatedDevice("pump", "masterflex")
        pump.connect()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            worker = ExperimentWorker({"pump": pump}, configs, [first, second], end_hold_s=0,
                                      recording={"enabled": True, "output_folder": root})
            accepted, idle_at_start = [], []
            worker.status_updated.connect(lambda packet: accepted.append(worker.request_skip(1))
                                          if not accepted else None)
            worker.experiment_changed.connect(lambda number, total: idle_at_start.append(not pump.is_running))
            worker.start_experiment()
            self.assertEqual(accepted, [True])
            self.assertEqual(idle_at_start, [True, True])
            self.assertEqual([run.status.value for run in worker.runs], ["skipped", "completed"])
            self.assertFalse(worker.request_skip(2))
            self.assertTrue(worker.recorder.complete.wait(5))
            coordinator = ExperimentCoordinator()
            controller = Controller(root, root, 2)
            coordinator.set_controller(controller)
            try:
                coordinator.begin_campaign(worker.campaign)
                coordinator.set_campaign_folder(worker.recorder.folder)
                coordinator.monitoring_tick()
                self.assertEqual(controller.current_stage, Stage.MODEL)
                model_input = json.loads((Path(controller.current_model_folder) / "campaign_input.json").read_text())
                self.assertEqual(model_input["experiments"], [worker.runs[1].experiment_id])
                self.assertEqual(model_input["skipped_experiments"], [worker.runs[0].experiment_id])
            finally:
                coordinator.reset()

    def test_dashboard_manual_loop_records_counts_and_enters_model_without_doe(self):
        from MAIN_WINDOW import MainWindow
        import time
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            model, experiments = Path(root) / "model", Path(root) / "experiments"
            model.mkdir()
            experiments.mkdir()
            settings = {"enabled": True, "sql_enabled": False, "output_folder": root, "remote_sql_url": ""}
            with patch("gui.recording_settings_dialog.load_recording_settings", return_value=settings):
                window = MainWindow()
                try:
                    window.control_panel.load_simulated_demo()
                    window.control_panel.hardware["TEM_MAIN"].connect()
                    window.control_panel.device_widgets["TEM_MAIN"]._update_connection_state()
                    window.settings_page.exp_path_input.setText(str(experiments))
                    window.settings_page.MODEL_path_input.setText(str(model))
                    window.settings_page.exp_number_input.clear()
                    window.control_panel.experiments = [pd.DataFrame({"time_s": [0.], "TEM_MAIN": [25.]})]
                    self.assertIsNone(window.create_experiments)
                    window.control_panel.simulation_speed.setValue(100)
                    with patch.object(window, "_choose_loop_source", return_value="manual"):
                        window.start_command()
                    self.assertIsNotNone(window.CONTROLLER)
                    deadline = time.monotonic() + 15
                    while window.CONTROLLER.current_stage != Stage.MODEL and time.monotonic() < deadline:
                        APP.processEvents()
                        time.sleep(.02)
                    self.assertEqual(window.CONTROLLER.current_stage, Stage.MODEL)
                    self.assertEqual(window.execution.conducted_experiments_count, 1)
                    self.assertTrue((Path(window.CONTROLLER.current_model_folder) / "campaign_input.json").exists())
                    from utilities.data_utils import StableResultTracker
                    now = [0.]
                    window.CONTROLLER.result_tracker = StableResultTracker(clock=lambda: now[0])
                    (Path(window.CONTROLLER.current_model_folder) / "MODEL.csv").write_text("completed\n")
                    window.execution.monitoring_tick()
                    now[0] = 3.
                    window.execution.monitoring_tick()
                    self.assertEqual(window.CONTROLLER.current_stage, Stage.OMNI)
                    self.assertTrue(window.execution.awaiting_campaign)
                    self.assertTrue(window.execution.timer.isActive())
                    self.assertEqual(window.dashboard_page.dashboard_omni.text(), "Running")
                    self.assertEqual(window.dashboard_page.dashboard_model.text(), "Inactive")
                    self.assertIn("monitoring active", window.dashboard_page.exp_status_box.title())
                    window.execution.monitoring_tick()
                    self.assertEqual(window.execution.model_run_count, 1)
                    window.stop_command()
                    APP.processEvents()
                    window.end_experiment_session()
                finally:
                    dashboard = window.experiment_page.dashboard
                    if dashboard is not None:
                        deadline = time.monotonic() + 10
                        while dashboard.thread is not None and time.monotonic() < deadline:
                            dashboard.stop_orchestrator()
                            APP.processEvents()
                            time.sleep(.02)
                        window.end_experiment_session()
                    window.close()
                    APP.processEvents()

    def fixture(self):
        configs = {"pump": {"type": "masterflex"}}
        definition = ExperimentDefinition.from_frame(pd.DataFrame({"time_s": [0., 1.], "pump": [10., 20.]}), configs)
        return configs, Campaign("Test", (definition,))

    def test_locked_csv_retries_without_failing_and_flushes_after_close(self):
        configs, campaign = self.fixture()
        locked = Event()
        locked.set()
        original_open = Path.open

        def file_open(path, *args, **kwargs):
            if path.name == "execution.csv" and locked.is_set():
                raise PermissionError("File is open in Excel")
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            with patch.object(Path, "open", file_open):
                recorder = RunRecorder(root, campaign, configs)
                try:
                    recorder.submit("run_start", "run1", {"planned": {"time_s": [0], "pump": [10]}})
                    for i in range(4):
                        recorder.submit("sample", "run1", {"elapsed_s": i, "pump.target": i})
                    recorder.submit("run_end", "run1", {"run_id": "run1", "experiment_id": campaign.experiments[0].experiment_id, "status": "completed"})
                    recorder.submit("summary", "", {"status": "completed"})
                    recorder.close()
                    self.assertIsNone(recorder.error)
                    self.assertTrue(recorder.waiting_for_csv)
                    self.assertFalse(recorder.complete.is_set())
                    self.assertFalse(json.loads((recorder.folder / "summary.json").read_text())["recording_complete"])
                    locked.clear()
                    self.assertTrue(recorder.complete.wait(5))
                    recorder.thread.join(2)
                    data = pd.read_csv(recorder.folder / "run1" / "execution.csv")
                    self.assertEqual(data["elapsed_s"].tolist(), [0, 1, 2, 3])
                    self.assertTrue(json.loads((recorder.folder / "run1" / "run.json").read_text())["recording_complete"])
                    self.assertFalse(recorder.database.exists())
                finally:
                    locked.clear()
                    recorder.thread.join(5)

    def test_campaign_count_ignores_csv_names_and_waits_for_complete_recordings(self):
        configs, campaign = self.fixture()
        coordinator = ExperimentCoordinator()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            model = Path(root) / "model"
            model.mkdir()
            controller = Controller(root, str(model), 999)
            coordinator.set_controller(controller)
            coordinator.begin_campaign(campaign)
            folder = Path(root) / "campaign_test"
            folder.mkdir()
            (folder / "definition.json").write_text(json.dumps({"campaign": {"campaign_id": campaign.campaign_id}}))
            coordinator.set_campaign_folder(folder)
            (folder / "expttsd1 - Copy.csv").write_text("ignored")
            run = folder / "run1"
            run.mkdir()
            marker = {"run_id": "run1", "experiment_id": campaign.experiments[0].experiment_id,
                      "status": "completed", "recording_complete": False}
            (run / "run.json").write_text(json.dumps(marker))
            (folder / "summary.json").write_text(json.dumps({"status": "completed", "recording_complete": False}))
            coordinator.monitoring_tick()
            self.assertEqual(coordinator.conducted_experiments_count, 0)
            self.assertEqual(controller.current_stage, Stage.OMNI)
            marker["recording_complete"] = True
            (run / "run.json").write_text(json.dumps(marker))
            (folder / "summary.json").write_text(json.dumps({"status": "completed", "recording_complete": True}))
            coordinator.monitoring_tick()
            self.assertEqual(coordinator.conducted_experiments_count, 1)
            self.assertEqual(controller.current_stage, Stage.MODEL)
            self.assertTrue((Path(controller.current_model_folder) / "campaign_input.json").exists())
            coordinator.reset()

    def test_csv_cannot_be_disabled_by_old_settings(self):
        self.assertTrue(validate_settings({"enabled": False, "output_folder": "runs"})["enabled"])

    def test_transient_campaign_reads_retry_and_preserve_counts(self):
        _, campaign = self.fixture()
        original_read = Path.read_text
        for name in ("definition.json", "run.json", "summary.json"):
            for failure in (PermissionError("OneDrive lock"), FileNotFoundError("Sync rename"),
                            json.JSONDecodeError("Partial sync", "{", 1)):
                with self.subTest(file=name, failure=type(failure).__name__), tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
                    coordinator = ExperimentCoordinator()
                    failures, progress = [], []
                    coordinator.error_occurred.connect(failures.append)
                    coordinator.experiment_progress.connect(progress.append)
                    controller = Controller(root, root, 1)
                    coordinator.set_controller(controller)
                    coordinator.begin_campaign(campaign)
                    folder = Path(root) / "campaign_test"
                    folder.mkdir()
                    (folder / "definition.json").write_text(json.dumps({"campaign": {"campaign_id": campaign.campaign_id}}))
                    run = folder / "run1"
                    run.mkdir()
                    (run / "run.json").write_text(json.dumps({"run_id": "run1", "experiment_id": campaign.experiments[0].experiment_id,
                                                             "status": "completed", "recording_complete": True}))
                    (folder / "summary.json").write_text(json.dumps({"status": "completed", "recording_complete": False}))
                    coordinator.set_campaign_folder(folder)
                    coordinator.monitoring_tick()
                    self.assertEqual(coordinator.conducted_experiments_count, 1)
                    (folder / "summary.json").write_text(json.dumps({"status": "completed", "recording_complete": True}))

                    def read(path, *args, **kwargs):
                        if path.name == name:
                            raise failure
                        return original_read(path, *args, **kwargs)

                    try:
                        with patch.object(Path, "read_text", read):
                            for _ in range(3):
                                coordinator.monitoring_tick()
                            self.assertEqual(controller.current_stage, Stage.OMNI)
                            self.assertFalse(controller.process_stop)
                            self.assertTrue(coordinator.timer.isActive())
                            self.assertEqual(coordinator.conducted_experiments_count, 1)
                        coordinator.monitoring_tick()
                        self.assertEqual(controller.current_stage, Stage.MODEL)
                        coordinator.monitoring_tick()
                        self.assertEqual(controller.omni_count, 1)
                        self.assertEqual(len(progress), 1)
                        self.assertEqual(failures, [])
                    finally:
                        coordinator.reset()

    def test_model_manifest_lock_retries_transition_with_existing_directory(self):
        _, campaign = self.fixture()
        coordinator = ExperimentCoordinator()
        failures = []
        coordinator.error_occurred.connect(failures.append)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            controller = Controller(root, root, 1)
            coordinator.set_controller(controller)
            coordinator.begin_campaign(campaign)
            folder = Path(root) / "campaign_test"
            folder.mkdir()
            (folder / "definition.json").write_text(json.dumps({"campaign": {"campaign_id": campaign.campaign_id}}))
            run = folder / "run1"
            run.mkdir()
            (run / "run.json").write_text(json.dumps({"run_id": "run1", "experiment_id": campaign.experiments[0].experiment_id,
                                                     "status": "completed", "recording_complete": True}))
            (folder / "summary.json").write_text(json.dumps({"status": "completed", "recording_complete": True}))
            coordinator.set_campaign_folder(folder)
            original = Path.write_text

            def write(path, *args, **kwargs):
                if path.name == "campaign_input.json":
                    raise PermissionError("Sync lock on manifest")
                return original(path, *args, **kwargs)

            try:
                with patch.object(Path, "write_text", write):
                    coordinator.monitoring_tick()
                    self.assertEqual(controller.current_stage, Stage.OMNI)
                    self.assertTrue(coordinator.timer.isActive())
                coordinator.monitoring_tick()
                self.assertEqual(controller.current_stage, Stage.MODEL)
                self.assertEqual(controller.omni_count, 1)
                self.assertEqual(failures, [])
            finally:
                coordinator.reset()

    def test_locked_run_metadata_retries_until_saved(self):
        from experiment.recording import atomic_json
        configs, campaign = self.fixture()
        locked, attempted = Event(), Event()
        locked.set()

        def publish(path, payload):
            if path.name == "run.json" and locked.is_set():
                attempted.set()
                raise PermissionError("OneDrive holds run marker")
            return atomic_json(path, payload)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root, patch("experiment.recording.atomic_json", publish):
            recorder = RunRecorder(root, campaign, configs)
            try:
                recorder.submit("run_start", "run1", {"planned": {"time_s": [0], "pump": [10]}})
                recorder.submit("sample", "run1", {"elapsed_s": 0, "pump.target": 10})
                recorder.submit("run_end", "run1", {"run_id": "run1", "experiment_id": campaign.experiments[0].experiment_id, "status": "completed"})
                recorder.submit("summary", "", {"status": "completed"})
                self.assertTrue(attempted.wait(5))
                recorder.close()
                self.assertIsNone(recorder.error)
                self.assertTrue(recorder.pending_csv)
                self.assertFalse(recorder.complete.is_set())
                locked.clear()
                self.assertTrue(recorder.complete.wait(5))
                self.assertTrue(json.loads((recorder.folder / "run1" / "run.json").read_text())["recording_complete"])
                self.assertTrue(json.loads((recorder.folder / "summary.json").read_text())["recording_complete"])
                self.assertEqual(len(pd.read_csv(recorder.folder / "run1" / "execution.csv")), 1)
            finally:
                locked.clear()
                recorder.thread.join(5)

    def test_completed_marker_is_not_rewritten_each_poll(self):
        import time
        from experiment.recording import atomic_json
        configs, campaign = self.fixture()
        published = Event()
        writes = []

        def publish(path, payload):
            result = atomic_json(path, payload)
            if path.name == "run.json" and payload.get("status") == "completed":
                writes.append(payload.copy())
                published.set()
            return result

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root, patch("experiment.recording.atomic_json", publish):
            recorder = RunRecorder(root, campaign, configs)
            try:
                recorder.submit("run_start", "run1", {"planned": {"time_s": [0], "pump": [10]}})
                recorder.submit("run_end", "run1", {"run_id": "run1", "experiment_id": campaign.experiments[0].experiment_id, "status": "completed"})
                self.assertTrue(published.wait(5))
                time.sleep(2.2)
                self.assertEqual(len(writes), 1)
            finally:
                recorder.close()

    def test_stopped_campaign_never_enters_model_or_reports_monitor_failure(self):
        _, campaign = self.fixture()
        coordinator = ExperimentCoordinator()
        failures = []
        coordinator.error_occurred.connect(failures.append)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            controller = Controller(root, root, 1)
            coordinator.set_controller(controller)
            coordinator.begin_campaign(campaign)
            folder = Path(root) / "campaign_test"
            folder.mkdir()
            (folder / "definition.json").write_text(json.dumps({"campaign": {"campaign_id": campaign.campaign_id}}))
            (folder / "summary.json").write_text(json.dumps({"status": "stopped", "recording_complete": True}))
            coordinator.set_campaign_folder(folder)
            coordinator.monitoring_tick()
            self.assertTrue(controller.process_stop)
            self.assertIsNone(controller.current_stage)
            self.assertEqual(failures, [])
            coordinator.reset()
