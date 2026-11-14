import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
import time
from experiment.models import ExperimentDefinition, ExperimentRun, RunStatus
from hardware_devices.simulated import SimulatedDevice, update_simulated_process
from hardware_devices.hardware_manager import HardwareManager, HardwareBusyError
from utilities.data_utils import counter, file_finder, StableResultTracker
from experiment.doe.doe_sampler import DoESampler
from experiment.experiment_service import ExperimentService
from experiment.execution_coordinator import ExperimentCoordinator
from app.controller import Controller, Stage
from experiment_dashboard.experiment_dashboard import ExperimentWorker, RealTimePlotCanvas

APP = QApplication.instance() or QApplication([])


class ModelTests(unittest.TestCase):
    def test_default_stirrer_accepts_entire_zero_to_800_range(self):
        from control_panel.control_panel import create_device, HARDWARE_TYPES
        config = {"type": "jkem", "name": "Stirrer", "connection": {"port": "unused"}}
        frame = pd.DataFrame({"time_s": list(range(801)), "stirrer": list(range(801))})
        ExperimentDefinition.from_frame(frame, {"stirrer": config})
        simulated = create_device({**config, "simulated": True}, HARDWARE_TYPES)
        simulated.connect()
        real = create_device(config, HARDWARE_TYPES)
        real._send_command = Mock(side_effect=lambda command: command)
        for rpm in range(801):
            self.assertTrue(simulated.set_target(rpm), rpm)
            self.assertTrue(real.set_target(rpm), rpm)
        for rpm in (-1, 801):
            self.assertFalse(simulated.set_target(rpm))
            self.assertFalse(real.set_target(rpm))
            with self.assertRaises(ValueError):
                ExperimentDefinition.from_frame(pd.DataFrame({"time_s": [0], "stirrer": [rpm]}), {"stirrer": config})

    def test_stirrer_override_reaches_validation_simulation_and_real_driver(self):
        from control_panel.control_panel import create_device, HARDWARE_TYPES
        config = {"type": "jkem", "name": "Stirrer", "connection": {"port": "unused"},
                  "target_limits": {"min": 0, "max": 1200, "min_nonzero": 0}}
        frame = pd.DataFrame({"time_s": [0., 1.], "stirrer": [5., 1100.]})
        ExperimentDefinition.from_frame(frame, {"stirrer": config})
        simulated = create_device({**config, "simulated": True}, HARDWARE_TYPES)
        simulated.connect()
        self.assertTrue(simulated.set_target(5))
        self.assertTrue(simulated.set_target(1100))
        real = create_device(config, HARDWARE_TYPES)
        real._send_command = Mock(side_effect=lambda command: command)
        self.assertTrue(real.set_target(5))
        self.assertTrue(real.set_target(1100))
        real._send_command.assert_called_with("SS1100")

    def test_stop_stays_available_with_positive_running_minimum(self):
        from hardware_devices.limits import target_limits, target_allowed
        limits = target_limits({"type": "jkem", "target_limits": {"min": 100, "max": 900}})
        self.assertTrue(target_allowed(0, limits))
        self.assertFalse(target_allowed(50, limits))

    def test_round_trip_and_distinct_runs(self):
        frame = pd.DataFrame({"time_s": [0, 1], "pump": [0, 10]})
        definition = ExperimentDefinition.from_frame(frame, {"pump": {"type": "masterflex"}})
        pd.testing.assert_frame_equal(definition.to_frame(), frame.astype(float))
        self.assertNotEqual(ExperimentRun(definition.experiment_id).run_id,
                            ExperimentRun(definition.experiment_id).run_id)

    def test_reject_invalid_profiles(self):
        config = {"pump": {"type": "masterflex"}}
        for times, targets in [([0, 0], [1, 2]), ([-1, 1], [1, 2]),
                               ([0, 1], [1, np.nan]), ([0, 1], [1, 301])]:
            with self.subTest(times=times, targets=targets), self.assertRaises(ValueError):
                ExperimentDefinition.from_frame(pd.DataFrame({"time_s": times, "pump": targets}), config)

    def test_valve_must_be_binary_and_reader_cannot_be_targeted(self):
        frame = pd.DataFrame({"time_s": [0], "device": [0.5]})
        for kind in ("nidaq_valve", "opcua", "mettler"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ExperimentDefinition.from_frame(frame, {"device": {"type": kind}})


class SimulationTests(unittest.TestCase):
    def test_all_device_contracts_without_io(self):
        for kind in sorted(SimulatedDevice.TYPES):
            with self.subTest(kind=kind):
                device = SimulatedDevice(kind, kind)
                self.assertTrue(device.connect())
                self.assertTrue(device.start())
                value = 1 if kind == "nidaq_valve" else 100 if kind == "jkem" else 10
                if kind not in {"mettler", "opcua"}:
                    self.assertTrue(device.set_target(value))
                self.assertTrue(device.read_status()["simulated"])
                self.assertTrue(device.stop())
                device.disconnect()
                with self.assertRaises(RuntimeError):
                    device.read_status()

    def test_repeatable_lag_and_noise(self):
        now = [0.0]
        devices = [SimulatedDevice("temperature", "huber", clock=lambda: now[0], seed=1) for _ in range(2)]
        for device in devices:
            device.connect()
            device.set_target(40)
        now[0] = 1
        readings = [d.read_status()["temp_internal"] for d in devices]
        self.assertEqual(readings[0], readings[1])
        self.assertGreater(readings[0], 25)
        self.assertLess(readings[0], 40)

    def test_flow_updates_simulated_balance(self):
        pump = SimulatedDevice("pump", "ismatec", noise=0)
        balance = SimulatedDevice("balance", "mettler", noise=0)
        pump.connect()
        balance.connect()
        pump.set_target(60)
        update_simulated_process({"p": pump, "b": balance}, {"p": {"type": "ismatec"}, "b": {"type": "mettler"}}, 2)
        self.assertEqual(balance.read_status()["weight_g"], 2)

    def test_fault_injection(self):
        device = SimulatedDevice("pump", "masterflex", fail_after=1)
        device.connect()
        self.assertTrue(device.set_target(10))
        self.assertFalse(device.set_target(20))


class ResultFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_ignore_backups_directories_and_partial_files(self):
        for name in ("expttsd1.csv", "expttsd_backup.csv", "expttsd2.csv.tmp", "expttsd3.csv.bak"):
            (self.folder / name).write_text("data")
        (self.folder / "expttsd4.csv").mkdir()
        self.assertEqual(counter(self.folder, "expttsd"), 1)
        self.assertEqual(file_finder(self.folder, "expttsd"), [str(self.folder / "expttsd1.csv")])

    def test_results_must_remain_nonempty_and_stable(self):
        now = [0.0]
        tracker = StableResultTracker(quiet_s=2, clock=lambda: now[0])
        result = self.folder / "expttsd1.csv"
        result.write_text("")
        self.assertEqual(tracker.files(self.folder, "expttsd"), [])
        now[0] = 3
        self.assertEqual(tracker.files(self.folder, "expttsd"), [])
        result.write_text("header\n1\n")
        self.assertEqual(tracker.files(self.folder, "expttsd"), [])
        now[0] = 6
        self.assertEqual(tracker.files(self.folder, "expttsd"), [result])
        result.write_text("header\n1\n2\n")
        self.assertEqual(tracker.files(self.folder, "expttsd"), [])

    def test_numbered_parentheses_and_windows_copy_names(self):
        names = ["expttsd1.csv", "expttsd(2).csv", "expttsd1 (2).csv", "expttsd_3.csv",
                 "expttsd1 - Copy.csv", "expttsd1 - Copy (2).csv",
                 "expttsd1 - Copy - Copy.csv", "expttsd1 - Copy - Copy (2).csv"]
        for name in names + ["expttsd_backup.csv", "expttsd1.csv.tmp"]:
            (self.folder / name).write_text("data")
        self.assertEqual(counter(self.folder, "expttsd"), len(names))

    def test_monitor_counts_immediately_and_does_not_recount_changed_file(self):
        now = [0.0]
        controller = Controller(str(self.folder), str(self.folder), 3)
        controller.result_tracker = StableResultTracker(clock=lambda: now[0])
        coordinator = ExperimentCoordinator()
        coordinator.set_controller(controller)
        coordinator.begin_omni_monitor(self.folder, 3)
        progress = []
        coordinator.experiment_progress.connect(progress.append)
        path = self.folder / "expttsd1 (2).csv"
        path.write_text("header\n1\n")
        coordinator.monitoring_tick()
        self.assertEqual(coordinator.conducted_experiments_count, 1)
        self.assertEqual(controller.completed_count(self.folder, "expttsd"), 0)
        now[0] = 3
        coordinator.monitoring_tick()
        path.write_text("header\n1\n2\n")
        coordinator.monitoring_tick()
        now[0] = 6
        coordinator.monitoring_tick()
        self.assertEqual(coordinator.conducted_experiments_count, 1)
        self.assertEqual(len(progress), 1)
        (self.folder / "expttsd2.csv").write_text("header\n1\n")
        coordinator.monitoring_tick()
        self.assertEqual(coordinator.conducted_experiments_count, 2)
        self.assertEqual(coordinator.monitor_mode, Stage.OMNI)
        coordinator.reset()

    def test_external_loop_counts_across_omni_model_omni(self):
        now = [0.0]
        omni_root, model_root = self.folder / "omni", self.folder / "model"
        omni_root.mkdir()
        model_root.mkdir()
        controller = Controller(str(omni_root), str(model_root), 2)
        controller.result_tracker = StableResultTracker(clock=lambda: now[0])
        batch, _ = controller.pathcreator_omni(0, str(omni_root))
        controller.current_omni_folder = batch
        coordinator = ExperimentCoordinator()
        coordinator.set_controller(controller)
        coordinator.begin_omni_monitor(batch, 2)
        for name in ["expttsd1.csv", "expttsd1 (2).csv"]:
            (Path(batch) / name).write_text("header\n1\n")
        coordinator.monitoring_tick()
        self.assertEqual(coordinator.conducted_experiments_count, 2)
        now[0] = 3
        coordinator.monitoring_tick()
        self.assertEqual(coordinator.monitor_mode, Stage.MODEL)
        (Path(controller.current_model_folder) / "MODEL1.mat").write_text("result")
        coordinator.monitoring_tick()
        now[0] = 6
        coordinator.monitoring_tick()
        self.assertEqual(coordinator.monitor_mode, Stage.OMNI)
        (Path(controller.current_omni_folder) / "expttsd1.csv").write_text("header\n1\n")
        coordinator.monitoring_tick()
        self.assertEqual(coordinator.conducted_experiments_count, 3)
        self.assertEqual(controller.model_count, 1)
        coordinator.reset()


class DoeTests(unittest.TestCase):
    def test_two_sample_generation(self):
        creator = ExperimentService().generate_initial_doe(2, "GREEDY")
        self.assertEqual(creator.actually_sampled_experiments, 2)
        self.assertEqual(len(creator.exp_storage.data), 2)

    def test_greedy_samples_are_unique(self):
        points = DoESampler().greedy_minimax(20, 3)
        self.assertEqual(len(np.unique(points, axis=0)), 20)

    def test_seven_factor_design(self):
        design = DoESampler().factorial_design(7)
        self.assertEqual(design.shape, (128, 7))
        self.assertEqual(len(np.unique(design, axis=0)), 128)

    def test_constant_dimension_scaling(self):
        result = DoESampler().scaler(np.array([[1., 2.], [1., 4.]]))
        np.testing.assert_array_equal(result, [[0, 0], [0, 1]])

    def test_invalid_factorial_rejected(self):
        for dims, fraction in [(11, 0), (0, 0), (3, 2), (4, 3)]:
            with self.subTest(dims=dims, fraction=fraction), self.assertRaises(ValueError):
                DoESampler().factorial_design(dims, fraction)

    def test_reexport_preserves_user_sheets_removes_stale_generated_sheets(self):
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            creator = ExperimentService().generate_initial_doe(1, "GREEDY")
            creator.export_OMNI_expts(folder)
            filename = Path(folder) / "agrprofiles.xlsx"
            workbook = load_workbook(filename)
            workbook.create_sheet("notes")["A1"] = "keep"
            workbook.create_sheet("subexp99")
            workbook.save(filename)
            workbook.close()
            creator.export_OMNI_expts(folder)
            workbook = load_workbook(filename)
            self.assertEqual(set(workbook.sheetnames), {"notes", "subexp1"})
            self.assertEqual(workbook.sheetnames[0], "subexp1")
            self.assertEqual(workbook["notes"]["A1"].value, "keep")
            workbook.close()

    def test_general_workbook_reexport_keeps_all_experiments(self):
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            creator = ExperimentService().generate_initial_doe(2, "GREEDY")
            filename = Path(folder) / "profiles.xlsx"
            creator.export_generated_expts(filename)
            creator.export_generated_expts(filename)
            workbook = load_workbook(filename)
            self.assertEqual(workbook.sheetnames, ["exp1", "exp2"])
            workbook.close()


class WorkerTests(unittest.TestCase):
    def make_worker(self, *, fail_after=None, **kwargs):
        device = SimulatedDevice("pump", "masterflex", fail_after=fail_after)
        device.connect()
        worker = ExperimentWorker({"pump": device}, {"pump": {"type": "masterflex"}},
                                  [pd.DataFrame({"time_s": [0.], "pump": [10.]})],
                                  end_hold_s=0, **kwargs)
        return worker, device

    def test_success_stops_pump_and_reports_completed(self):
        worker, device = self.make_worker()
        result = []
        worker.run_finished.connect(result.append)
        worker.start_experiment()
        self.assertEqual(result[0]["status"], "completed")
        self.assertFalse(device.is_running)
        self.assertTrue(device.is_connected)
        self.assertEqual(worker.runs[0].status, RunStatus.COMPLETED)

    def test_rejected_command_reports_failure(self):
        worker, device = self.make_worker(fail_after=0)
        worker.start_experiment()
        self.assertEqual(worker.outcome, RunStatus.FAILED)
        self.assertIn("rejected", worker.error_message)
        self.assertFalse(device.is_running)

    def test_stop_before_thread_started_is_preserved(self):
        worker, device = self.make_worker()
        worker.request_stop()
        worker.start_experiment()
        self.assertEqual(worker.outcome, RunStatus.STOPPED)
        self.assertFalse(device.is_running)

    def test_command_timeout_aborts(self):
        worker, device = self.make_worker()
        future = Mock()
        future.result.side_effect = TimeoutError()
        worker.executor = Mock()
        worker.executor.submit.return_value = future
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            worker._apply_profile_step(pd.Series({"time_s": 0, "pump": 10}))

    def test_cleanup_failure_is_not_success(self):
        worker, device = self.make_worker()
        device.stop = Mock(return_value=False)
        worker.start_experiment()
        self.assertEqual(worker.outcome, RunStatus.FAILED)
        self.assertIn("Safe-idle", worker.error_message)

    def test_real_devices_cannot_run_accelerated(self):
        device = Mock(simulated=False)
        with self.assertRaisesRegex(ValueError, "Accelerated"):
            ExperimentWorker({"pump": device}, {"pump": {"type": "masterflex"}},
                             [pd.DataFrame({"time_s": [0], "pump": [0]})], speed=10)

    def test_discrete_valve_interpolation(self):
        config = {"valve": {"type": "nidaq_valve"}}
        profile = pd.DataFrame({"time_s": [0., 10.], "valve": [0., 1.]})
        worker = ExperimentWorker({"valve": SimulatedDevice("valve", "nidaq_valve")}, config, [profile])
        self.assertEqual(worker._interpolate_profile_row(profile, 9)["valve"], 0)
        self.assertEqual(worker._interpolate_profile_row(profile, 10)["valve"], 1)


class CoordinatorTests(unittest.TestCase):
    def test_model_completion_continues_external_folder_loop(self):
        coordinator = ExperimentCoordinator()
        controller = Controller("unused", "unused", 1)
        controller.current_model_folder = "unused/model"
        controller.if_model_finished = Mock(return_value=True)
        controller.pathcreator_omni = Mock(return_value=("unused/omni1", 1))
        coordinator.set_controller(controller)
        coordinator.begin_model_monitor("unused/model")
        result = coordinator.check_model_transition("unused")
        self.assertTrue(result["external_experiment_required"])
        controller.pathcreator_omni.assert_called_once_with(1, "unused")
        self.assertTrue(coordinator.timer.isActive())
        self.assertEqual(coordinator.monitor_mode, Stage.OMNI)
        self.assertEqual(controller.current_omni_folder, "unused/omni1")
        coordinator.reset()

    def test_provider_exports_next_experiment(self):
        coordinator = ExperimentCoordinator()
        controller = Controller("omni", "model", 1)
        controller.current_model_folder = "model/0"
        controller.if_model_finished = Mock(return_value=True)
        controller.pathcreator_omni = Mock(return_value=("omni/1", 1))
        creator = Mock(actually_sampled_experiments=3)
        coordinator.next_experiment_provider = Mock(return_value=creator)
        coordinator.set_controller(controller)
        result = coordinator.check_model_transition("omni")
        creator.export_OMNI_expts.assert_called_once_with("omni/1")
        self.assertEqual(result["exp_count"], 3)
        self.assertEqual(coordinator.monitor_mode, Stage.OMNI)
        coordinator.reset()

    def test_export_failure_stops_execution(self):
        coordinator = ExperimentCoordinator()
        controller = Controller("omni", "model", 1)
        controller.pathcreator_omni = Mock(return_value=("omni/0", 1))
        coordinator.set_controller(controller)
        coordinator.set_data_processor(Mock())
        creator = Mock()
        creator.export_OMNI_expts.side_effect = OSError("disk full")
        coordinator.set_experiment_creator(creator)
        with self.assertRaises(OSError):
            coordinator.start_execution("omni", 1)
        self.assertTrue(controller.process_stop)
        self.assertFalse(coordinator.timer.isActive())


class ControllerTests(unittest.TestCase):
    def test_start_stop_reset(self):
        controller = Controller("omni", "model", 3)
        controller.start()
        self.assertEqual(controller.current_stage, Stage.OMNI)
        controller.stop()
        self.assertTrue(controller.process_stop)
        controller.reset()
        self.assertIsNone(controller.current_stage)
        self.assertFalse(controller.process_stop)

    def test_ownership_requires_matching_token(self):
        manager = HardwareManager({})
        token = manager.reserve_experiment()
        with self.assertRaises(HardwareBusyError):
            manager.reserve_connection()
        self.assertFalse(manager.release_experiment("wrong"))
        self.assertTrue(manager.owns_experiment(token))
        self.assertTrue(manager.release_experiment(token))
        self.assertFalse(manager.is_busy)


class GuiTests(unittest.TestCase):
    def setUp(self):
        settings = {"enabled": True, "sql_enabled": False, "output_folder": str(Path.cwd() / "data" / "runs"), "remote_sql_url": ""}
        mocked = patch("gui.recording_settings_dialog.load_recording_settings", return_value=settings)
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_imported_main_window_and_dialog_share_cached_application_icon(self):
        from MAIN_WINDOW import MainWindow
        from PySide6.QtWidgets import QDialog
        from app.branding import configure_application_icon
        window = MainWindow()
        dialog = QDialog()
        try:
            self.assertFalse(window.windowIcon().isNull())
            self.assertEqual(window.windowIcon().cacheKey(), APP.windowIcon().cacheKey())
            self.assertEqual(dialog.windowIcon().cacheKey(), APP.windowIcon().cacheKey())
            window.show()
            APP.processEvents()
            self.assertFalse(window.windowHandle().icon().isNull())
            self.assertEqual(window.windowHandle().icon().cacheKey(), APP.windowIcon().cacheKey())
            window.hide()
            window.show()
            APP.processEvents()
            self.assertFalse(window.windowHandle().icon().isNull())
            # Later icon requests do not reopen the original synchronized asset.
            with patch("app.branding.LOGO_PATH", Path("missing-logo.png")):
                configure_application_icon(APP, window)
                for size in (16, 32, 48, 256):
                    pixmap = window.windowIcon().pixmap(size, size)
                    self.assertFalse(pixmap.isNull())
                    self.assertEqual(pixmap.width(), size)
                    self.assertEqual(pixmap.height(), size)
        finally:
            dialog.close()
            window.close()
            APP.processEvents()

    def test_slow_renderer_does_not_block_gui_or_queue_unbounded_frames(self):
        from threading import Event
        from PySide6.QtCore import QThread, QTimer
        from gui.plot_renderer import OriginalPlotRenderer
        entered, release = Event(), Event()
        render_threads, requests, ticks = [], [], []

        def delayed_renderer(*args, **kwargs):
            render_threads.append(QThread.currentThread())
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test renderer was not released")
            return OriginalPlotRenderer(*args, **kwargs)

        canvas = RealTimePlotCanvas(device_configs={"pump": {"type": "masterflex"}})
        timer = QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: ticks.append(True))
        canvas.render_requested.connect(requests.append)
        try:
            with patch("gui.live_plot_widget.OriginalPlotRenderer", side_effect=delayed_renderer):
                canvas.append_and_update({"timestamp": 0, "pump": {"speed_rpm": 0}})
                canvas.flush_pending()
                self.wait_for(entered.is_set)
                timer.start()
                self.wait_for(lambda: len(ticks) >= 3)
                self.assertFalse(release.is_set())
                self.assertIsNot(render_threads[0], APP.thread())
                for i in range(1, 501):
                    canvas.append_and_update({"timestamp": i, "pump": {"speed_rpm": i}})
                    canvas.flush_pending()
                self.assertEqual(len(requests), 1)
                release.set()
                self.wait_for(lambda: len(requests) == 2 and not canvas._rendering)
                self.assertEqual(requests[-1]["times"][-1], 500)
                self.assertFalse(canvas._image.isNull())
                self.assertFalse(canvas.render_error)
        finally:
            release.set()
            timer.stop()
            self.assertTrue(canvas.shutdown_renderer())
            self.assertIsNone(canvas._thread)
            canvas.close()

    def test_frame_from_previous_experiment_is_ignored(self):
        from PySide6.QtGui import QImage
        canvas = RealTimePlotCanvas(device_configs={"pump": {"type": "masterflex"}})
        try:
            canvas._refresh_timer.stop()
            previous = canvas._generation
            canvas.set_experiment_title(2, 2)
            image = QImage(20, 20, QImage.Format.Format_RGBA8888)
            image.fill(0)
            canvas._accept_frame(previous, image)
            self.assertTrue(canvas._image.isNull())
            canvas._accept_frame(canvas._generation, image)
            self.assertFalse(canvas._image.isNull())
        finally:
            canvas.close()

    def test_original_plot_assignments_and_secondary_axes_are_restored(self):
        configs = {
            "stirrer": {"name": "Stirrer", "type": "jkem", "plot_index": 1},
            "temperature": {"name": "Temperature", "type": "huber", "plot_index": 4},
            "valve": {"name": "Valve", "type": "nidaq_valve", "plot_index": 5},
        }
        from gui.plot_renderer import OriginalPlotRenderer
        canvas = OriginalPlotRenderer(device_configs=configs)
        try:
            self.assertEqual(len(canvas.axes), 8)
            self.assertTrue(all(axis.get_title() == "" for axis in canvas.axes))
            self.assertIs(canvas.lines[("temperature", "setpoint")].axes, canvas.axes[3])
            self.assertIs(canvas.secondary_lines[("stirrer", "torque_ozin")].axes, canvas.secondary_axes[1])
            self.assertEqual(canvas.secondary_axes[5].get_ylim(), (-0.05, 1.05))
            self.assertEqual(canvas.secondary_lines[("valve", "target_state")].get_linestyle(), "-.")
            legend = [label.get_text() for label in canvas.axes[0].get_legend().get_texts()]
            self.assertIn("Stirrer", legend)
            self.assertIn("Stirrer - Torque", legend)
        finally:
            canvas.fig.clear()

    def test_live_plot_paints_in_both_themes(self):
        canvas = RealTimePlotCanvas(device_configs={"pump": {"type": "masterflex"}})
        canvas.resize(1000, 1000)
        try:
            canvas.append_and_update({"timestamp": 0, "pump": {"speed_rpm": 10}})
            canvas.flush_pending()
            canvas.append_and_update({"timestamp": 1, "pump": {"speed_rpm": 20}})
            canvas.flush_pending()
            for dark in (True, False):
                canvas.set_theme(dark)
                canvas.flush_pending()
                self.wait_for(lambda: not canvas._rendering and not canvas._image.isNull())
                self.assertFalse(canvas.render_error)
                self.assertFalse(canvas.grab().isNull())
        finally:
            canvas._refresh_timer.stop()
            canvas.close()

    def test_limits_editor_applies_and_configuration_retains_overrides(self):
        from control_panel.control_panel import ControlPanel, HARDWARE_TYPES, create_hardware, DeviceLimitsDialog
        configs = {"stirrer": {"type": "jkem", "name": "Demo", "connection": {}, "simulated": True}}
        panel = ControlPanel(configs, HARDWARE_TYPES, create_hardware(configs, HARDWARE_TYPES), configs)
        try:
            dialog = DeviceLimitsDialog(configs, panel)
            dialog.rows["stirrer"][1]["max"].setValue(1200)
            dialog.rows["stirrer"][1]["min_nonzero"].setValue(0)
            panel.apply_device_limits(dialog.overrides())
            device = panel.hardware["stirrer"]
            device.connect()
            self.assertTrue(device.set_target(5))
            saved, _ = panel.collect_current_configuration()
            self.assertEqual(saved["stirrer"]["target_limits"]["max"], 1200)
            self.assertEqual(saved["stirrer"]["target_limits"]["min_nonzero"], 0)
            self.assertEqual(panel.device_widgets["stirrer"].target_widget.maximum(), 1200)
            dialog.close()
        finally:
            panel.close()

    def test_telemetry_burst_preserves_samples_and_coalesces_rendering(self):
        canvas = RealTimePlotCanvas(device_configs={"pump": {"type": "masterflex"}})
        try:
            with patch.object(canvas, "_dispatch_render") as draw:
                for i in range(500):
                    canvas.append_and_update({"timestamp": i, "pump": {"speed_rpm": i}})
                self.assertEqual(len(canvas.t_data), 0)
                draw.assert_not_called()
                canvas.flush_pending()
                self.assertEqual(list(canvas.t_data), list(range(500)))
                draw.assert_called_once()
        finally:
            canvas._refresh_timer.stop()
            canvas.close()

    def test_plot_history_is_bounded_and_axes_reused_between_experiments(self):
        from collections import deque
        canvas = RealTimePlotCanvas(device_configs={"pump": {"type": "masterflex"}})
        try:
            history_keys = tuple(canvas.history)
            canvas.t_data = deque(maxlen=3)
            canvas.history = {key: deque(maxlen=3) for key in canvas.history}
            with patch.object(canvas, "_dispatch_render"):
                for i in range(5):
                    canvas.append_and_update({"timestamp": i, "pump": {"speed_rpm": i}})
                    canvas.flush_pending()
                self.assertEqual(list(canvas.t_data), [2, 3, 4])
                self.assertEqual(list(canvas.history[("pump", "speed_rpm")]), [2, 3, 4])
                canvas.set_experiment_title(2, 3)
                self.assertEqual(tuple(canvas.history), history_keys)
                self.assertEqual(len(canvas.t_data), 0)
                self.assertIsNone(canvas._pending_data)
        finally:
            canvas._refresh_timer.stop()
            canvas.close()

    def wait_for(self, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(0.02)
        self.assertTrue(predicate(), "GUI operation did not finish before timeout")

    def test_full_demo_connect_execute_end_session_and_close(self):
        from MAIN_WINDOW import MainWindow
        window = MainWindow()
        panel = window.control_panel
        try:
            panel.load_simulated_demo()
            self.assertEqual(panel.simulation_speed.value(), 1.0)
            panel.simulation_speed.setValue(10)  # Explicit acceleration for this automated test only.
            self.assertEqual(len(panel.experiments), 1)
            self.assertTrue(all(getattr(d, "simulated", False) for d in panel.hardware.values()))
            panel.connect_all()
            self.wait_for(lambda: panel.connection_thread is None)
            self.assertTrue(all(d.is_connected for d in panel.hardware.values()))
            panel.start_experiment()
            dashboard = window.experiment_page.dashboard
            self.assertIsNotNone(dashboard)
            self.assertEqual(dashboard.recording_settings["output_folder"], panel.control_output_path.text())
            self.assertNotEqual(dashboard.recording_settings["output_folder"], str(Path.cwd() / "data" / "runs"))
            # Reserved startup is active too, before the 200 ms start timer fires.
            self.assertTrue(dashboard.is_execution_active())
            self.assertFalse(window.experiment_page.end_session_button.isEnabled())
            window.end_experiment_session()
            self.assertIs(window.experiment_page.dashboard, dashboard)
            self.assertFalse(dashboard.close())
            self.wait_for(lambda: dashboard.thread is not None)
            with patch.object(dashboard.worker, "request_stop", wraps=dashboard.worker.request_stop) as stop:
                window.end_experiment_session()
                self.assertFalse(dashboard.close())
                stop.assert_not_called()
            self.wait_for(lambda: hasattr(dashboard, "last_result") and dashboard.thread is None)
            self.assertEqual(dashboard.last_result["status"], "completed")
            self.assertFalse(panel.hardware_manager.is_busy)
            self.assertTrue(dashboard.btn_start.isEnabled())
            self.assertTrue(window.experiment_page.end_session_button.isEnabled())
            # Restart and stop the same dashboard; no stale QThread may block it.
            del dashboard.last_result
            dashboard.start_orchestrator()
            self.assertFalse(window.experiment_page.end_session_button.isEnabled())
            APP.processEvents()
            time.sleep(0.1)
            dashboard.stop_orchestrator()
            self.assertFalse(window.experiment_page.end_session_button.isEnabled())
            window.end_experiment_session()
            self.assertIs(window.experiment_page.dashboard, dashboard)
            self.wait_for(lambda: hasattr(dashboard, "last_result") and dashboard.thread is None)
            self.assertEqual(dashboard.last_result["status"], "stopped")
            self.assertTrue(window.experiment_page.end_session_button.isEnabled())
            window.end_experiment_session()
            self.assertIsNone(window.experiment_page.dashboard)
            self.assertIsNone(panel.experiment_window)
        finally:
            dashboard = window.experiment_page.dashboard
            if dashboard is not None:
                dashboard.stop_orchestrator()
                self.wait_for(lambda: dashboard.thread is None)
                window.end_experiment_session()
            window.close()
            APP.processEvents()

    def test_new_simulated_device_configuration(self):
        from control_panel.control_panel import NewDeviceDialog, HARDWARE_TYPES, create_device
        dialog = NewDeviceDialog(HARDWARE_TYPES)
        dialog.id_edit.setText("SIM_PUMP")
        dialog.name_edit.setText("Demo pump")
        dialog.mode_combo.setCurrentText("Simulated")
        device_id, config = dialog.get_device()
        self.assertEqual(device_id, "SIM_PUMP")
        device = create_device(config, HARDWARE_TYPES)
        self.assertIsInstance(device, SimulatedDevice)
        self.assertFalse(device.is_connected)
        dialog.close()
