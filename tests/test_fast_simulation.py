import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PySide6.QtWidgets import QApplication
from experiment_dashboard.experiment_dashboard import ExperimentWorker
from experiment.models import ExperimentDefinition, DeviceProfile, ProfilePoint
from hardware_devices.simulated import SimulatedDevice
from gui.live_plot_widget import LivePlotWidget

APP = QApplication.instance() or QApplication([])


class FastSimulationTests(unittest.TestCase):
    def test_simulation_trace_is_independent_of_acceleration_factor(self):
        traces = []
        for speed in (10, 100):
            pump = SimulatedDevice("pump", "masterflex", noise=0)
            pump.connect()
            frame = pd.DataFrame({"time_s": [0., .01, .02, .1], "pump": [0., 50., 0., 0.]})
            worker = ExperimentWorker({"pump": pump}, {"pump": {"type": "masterflex"}}, [frame],
                                      speed=speed, end_hold_s=0)
            packets = []
            worker.status_updated.connect(packets.append)
            worker.start_experiment()
            self.assertEqual(worker.outcome.value, "completed", worker.error_message)
            traces.append([(sample["timestamp"], sample["pump"]["speed_rpm"])
                           for packet in packets for sample in packet["samples"]])
        self.assertEqual(traces[0], traces[1])

    def test_failed_accelerated_run_restores_device_clock(self):
        pump = SimulatedDevice("pump", "masterflex", fail_after=2)
        pump.connect()
        original = pump._clock
        worker = ExperimentWorker({"pump": pump}, {"pump": {"type": "masterflex"}},
            [pd.DataFrame({"time_s": [0., .1], "pump": [0., 50.]})], speed=100, end_hold_s=0)
        worker.start_experiment()
        self.assertEqual(worker.outcome.value, "failed")
        self.assertIs(pump._clock, original)
        self.assertEqual(pump.time_scale, 1)
        self.assertFalse(pump.is_running)

    def test_100x_records_short_pulse_and_delivers_it_to_plot_in_batches(self):
        pump = SimulatedDevice("pump", "masterflex", noise=0, response_time=.0001)
        pump.connect()
        original_clock = pump._clock
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            frame = pd.DataFrame({"time_s": [0., .01, .02, .1], "pump": [0., 50., 0., 0.]})
            worker = ExperimentWorker({"pump": pump}, {"pump": {"type": "masterflex"}}, [frame],
                speed=100, poll_interval=2, end_hold_s=0,
                recording={"enabled": True, "output_folder": root})
            packets = []
            worker.status_updated.connect(packets.append)
            worker.start_experiment()
            self.assertEqual(worker.outcome.value, "completed", worker.error_message)
            recorded = pd.read_csv(worker.recorder.folder / worker.runs[0].run_id / "execution.csv")
            self.assertAlmostEqual(recorded["pump.target"].max(), 50)
            self.assertGreaterEqual(len(recorded[recorded.elapsed_s.between(0, .02)]), 17)
            self.assertEqual(recorded.elapsed_s.iloc[-1], .1)
            self.assertIs(pump._clock, original_clock)
            self.assertEqual(pump.time_scale, 1)
            self.assertFalse(pump.is_running)
            canvas = LivePlotWidget(device_configs={"pump": {"type": "masterflex"}})
            try:
                with patch.object(canvas, "_dispatch_render") as draw:
                    for packet in packets:
                        canvas.append_and_update(packet)
                    canvas.flush_pending()
                    self.assertEqual(len(canvas.t_data), len(recorded))
                    self.assertGreater(max(canvas.history[("pump", "speed_rpm")]), 40)
                    draw.assert_called_once()
            finally:
                canvas.close()

    def test_short_step_pulse_keeps_correct_simulated_dose(self):
        pump = SimulatedDevice("pump", "ismatec", noise=0)
        scale = SimulatedDevice("scale", "mettler", noise=0)
        for device in (pump, scale):
            device.connect()
        definition = ExperimentDefinition("Pulse", (DeviceProfile("pump", (
            ProfilePoint(0, 60), ProfilePoint(.2, 0), ProfilePoint(.3, 0)), "step", "mL/min"),))
        worker = ExperimentWorker({"pump": pump, "scale": scale},
            {"pump": {"type": "ismatec"}, "scale": {"type": "mettler"}}, [definition], speed=100, end_hold_s=0)
        worker.start_experiment()
        self.assertEqual(worker.outcome.value, "completed", worker.error_message)
        self.assertAlmostEqual(scale.weight, .2, places=8)

    def test_bounded_full_experiment_history_retains_early_short_peak(self):
        canvas = LivePlotWidget(device_configs={"pump": {"type": "masterflex"}})
        try:
            with patch.object(canvas, "_dispatch_render"):
                for begin in range(0, 10000, 100):
                    samples = tuple({"timestamp": i, "profile_boundary": i in (99, 100, 101),
                        "pump": {"speed_rpm": 50 if i == 100 else 0}} for i in range(begin, begin + 100))
                    canvas.append_and_update({"samples": samples})
                    canvas.flush_pending()
                self.assertLessEqual(len(canvas.t_data), 2000)
                self.assertEqual(canvas.t_data[0], 0)
                self.assertEqual(canvas.t_data[-1], 9999)
                self.assertIn(100, canvas.t_data)
                self.assertEqual(max(canvas.history[("pump", "speed_rpm")]), 50)
                self.assertEqual(len(canvas.t_data), len(canvas.history[("pump", "speed_rpm")]))
        finally:
            canvas.close()

    def test_pending_backlog_is_bounded_without_losing_peak(self):
        canvas = LivePlotWidget(device_configs={"pump": {"type": "masterflex"}})
        try:
            with patch.object(canvas, "_dispatch_render"):
                for i in range(10000):
                    canvas.append_and_update({"timestamp": i, "profile_boundary": i == 42,
                                              "pump": {"speed_rpm": 50 if i == 42 else 0}})
                self.assertLessEqual(len(canvas._pending_samples), 4096)
                canvas.flush_pending()
                self.assertEqual(max(canvas.history[("pump", "speed_rpm")]), 50)
                self.assertEqual(canvas.t_data[-1], 9999)
        finally:
            canvas.close()
