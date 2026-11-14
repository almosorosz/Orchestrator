import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
import pandas as pd
from experiment.models import Campaign, ExperimentDefinition
from experiment.recording import RunRecorder, sync_sql, validate_settings
from experiment_dashboard.experiment_dashboard import ExperimentWorker
from hardware_devices.simulated import SimulatedDevice


class RecordingTests(unittest.TestCase):
    def test_manual_and_sampled_batches_use_ordered_unique_run_ids(self):
        from uuid import UUID
        configs = {"pump": {"type": "masterflex"}}
        device = SimulatedDevice("pump", "masterflex")
        frame = pd.DataFrame({"time_s": [0.], "pump": [10.]})
        for sampled in (False, True):
            definitions = tuple(ExperimentDefinition.from_frame(frame, configs, name=f"Experiment {i}", source="doe") for i in (1, 2))
            experiments = list(definitions) if sampled else [frame, frame.copy()]
            campaign = Campaign("Sampled", definitions) if sampled else None
            worker = ExperimentWorker({"pump": device}, configs, experiments, campaign=campaign)
            rerun = ExperimentWorker({"pump": device}, configs, experiments, campaign=campaign)
            for number, run in enumerate(worker.runs, 1):
                prefix, identifier = run.run_id.split("_", 1)
                self.assertEqual(int(prefix), number)
                UUID(identifier)
                self.assertNotEqual(run.run_id, rerun.runs[number - 1].run_id)

    def test_torque_is_recorded_only_for_stirrers_in_csv_and_sql(self):
        configs = {"pump": {"type": "masterflex"}, "ismatec": {"type": "ismatec"}, "stirrer": {"type": "jkem"}}
        definition = ExperimentDefinition.from_frame(pd.DataFrame({"time_s": [0.], "pump": [10.], "stirrer": [100.]}), configs)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            recorder = RunRecorder(root, Campaign("Torque schema", (definition,)), configs, {"sql_enabled": True})
            recorder.submit("run_start", "run1", {"planned": {"time_s": [0], "pump": [10], "stirrer": [100]}})
            recorder.submit("sample", "run1", {"elapsed_s": 0, "pump.speed_rpm": 10, "pump.torque_ozin": 1,
                "ismatec.flow": 2, "ismatec.torque_ozin": 2, "stirrer.torque_ozin": 3})
            recorder.close()
            frame = pd.read_csv(recorder.folder / "run1" / "execution.csv")
            self.assertNotIn("pump.torque_ozin", frame)
            self.assertNotIn("ismatec.torque_ozin", frame)
            self.assertEqual(frame["stirrer.torque_ozin"].tolist(), [3])
            with closing(sqlite3.connect(recorder.database)) as db:
                payload = json.loads(db.execute("SELECT payload FROM events WHERE kind='sample'").fetchone()[0])
            self.assertNotIn("pump.torque_ozin", payload)
            self.assertNotIn("ismatec.torque_ozin", payload)
            self.assertEqual(payload["stirrer.torque_ozin"], 3)
            from experiment.archive import plot_traces
            # Historical files can still contain pump torque: omit those traces.
            legacy = recorder.folder / "legacy"
            legacy.mkdir()
            pd.DataFrame({"elapsed_s": [0], "pump.torque_ozin": [1], "stirrer.torque_ozin": [3]}).to_csv(legacy / "execution.csv", index=False)
            traces = plot_traces({"folder": str(legacy)}, device_configs=configs)
            self.assertNotIn("pump.torque_ozin", traces)
            self.assertIn("stirrer.torque_ozin", traces)

    def test_csv_only_creates_no_sql_database(self):
        configs = {"pump": {"type": "masterflex"}}
        definition = ExperimentDefinition.from_frame(pd.DataFrame({"time_s": [0.], "pump": [10.]}), configs)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            recorder = RunRecorder(root, Campaign("CSV only", (definition,)), configs, {"sql_enabled": False})
            recorder.submit("run_start", "run1", {"planned": {"time_s": [0], "pump": [10]}})
            recorder.submit("sample", "run1", {"elapsed_s": 0, "pump.target": 10})
            recorder.close()
            self.assertTrue((recorder.folder / "run1" / "execution.csv").exists())
            self.assertFalse(recorder.database.exists())

    def test_continuous_csv_sql_outbox_and_offline_recovery(self):
        configs = {"pump": {"type": "masterflex", "simulated": True}}
        definition = ExperimentDefinition.from_frame(pd.DataFrame({"time_s": [0.], "pump": [10.]}), configs)
        campaign = Campaign("Test", (definition,))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            recorder = RunRecorder(root, campaign, configs, {"sql_enabled": True})
            recorder.submit("run_start", "run1", {"planned": {"time_s": [0], "pump": [10]}})
            for i in range(3):
                recorder.submit("sample", "run1", {"elapsed_s": i, "pump.target": 10, "pump.speed_rpm": i})
            recorder.submit("summary", "", {"status": "completed"})
            recorder.close()
            from gui.recording_settings_dialog import database_counts
            self.assertEqual(database_counts(recorder.database), (6, 6))
            data = pd.read_csv(recorder.folder / "run1" / "execution.csv")
            self.assertEqual(data["elapsed_s"].tolist(), [0, 1, 2])
            self.assertEqual(data["pump.target"].tolist(), [10, 10, 10])
            self.assertIn("utc", data)
            with self.assertRaises(sqlite3.OperationalError):
                sync_sql(recorder.database, "sqlite:///" + str(Path(root) / "absent" / "cloud.db"))
            with closing(sqlite3.connect(recorder.database)) as db:
                count = db.execute("SELECT count(*) FROM events WHERE uploaded=0").fetchone()[0]
            self.assertEqual(count, 6)
            remote = Path(root) / "cloud.db"
            self.assertEqual(sync_sql(recorder.database, "sqlite:///" + str(remote)), count)
            self.assertEqual(sync_sql(recorder.database, "sqlite:///" + str(remote)), 0)
            with closing(sqlite3.connect(recorder.database)) as db:
                db.execute("UPDATE events SET uploaded=0")
                db.commit()
            sync_sql(recorder.database, "sqlite:///" + str(remote))
            with closing(sqlite3.connect(remote)) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM events").fetchone()[0], count)

    def test_worker_records_actual_samples_and_final_outcome(self):
        device = SimulatedDevice("pump", "masterflex")
        device.connect()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            worker = ExperimentWorker({"p": device}, {"p": {"type": "masterflex"}},
                                      [pd.DataFrame({"time_s": [0.], "p": [10.]})], end_hold_s=0,
                                      recording={"enabled": True, "output_folder": root})
            worker.start_experiment()
            self.assertEqual(worker.outcome.value, "completed")
            folder = worker.recorder.folder
            self.assertTrue(worker.runs[0].run_id.startswith("1_"))
            sample = pd.read_csv(folder / worker.runs[0].run_id / "execution.csv")
            self.assertEqual(sample["p.target"].iloc[0], 10.)
            self.assertNotIn("p.torque_ozin", sample)
            self.assertNotIn("torque_ozin", device.read_status())
            self.assertEqual(json.loads((folder / "summary.json").read_text())["status"], "completed")
            marker = json.loads((folder / worker.runs[0].run_id / "run.json").read_text())
            self.assertEqual(marker["experiment_number"], 1)
            self.assertEqual(marker["batch_name"], worker.runs[0].run_id)
            self.assertFalse(worker.recorder.thread.is_alive())

    def test_old_unit_definitions_are_ignored(self):
        settings = {"output_folder": "runs", "unit_operations": [{"code": "CR01", "name": "Main reactor", "kind": "crystallizer"}],
                    "device_units": {"probe": "CR01"}}
        cleaned = validate_settings(settings)
        self.assertNotIn("unit_operations", cleaned)
        self.assertNotIn("device_units", cleaned)
        self.assertEqual(cleaned["output_folder"], "runs")

    def test_sql_settings_demo_and_status_without_unit_controls(self):
        from unittest.mock import patch
        from PySide6.QtWidgets import QApplication
        from gui.recording_settings_dialog import RecordingSettingsDialog, load_recording_settings, database_counts
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            settings_file = Path(folder) / "settings.json"
            with patch("gui.recording_settings_dialog.SETTINGS_FILE", settings_file):
                dialog = RecordingSettingsDialog()
                try:
                    self.assertFalse(hasattr(dialog, "units"))
                    dialog.folder.setText(folder)
                    dialog.refresh_status()
                    self.assertIn("not created yet", dialog.status.text())
                    self.assertIsNone(database_counts(Path(folder) / "recordings.sqlite3"))
                    with patch("gui.recording_settings_dialog.QFileDialog.getExistingDirectory", return_value=folder):
                        dialog.choose_demo()
                    self.assertTrue(dialog.sql_enabled.isChecked())
                    self.assertTrue(dialog.sql.text().endswith("/cloud_demo.sqlite3"))
                    dialog.save()
                    saved = load_recording_settings()
                    self.assertNotIn("unit_operations", saved)
                    self.assertTrue(saved["sql_enabled"])
                finally:
                    dialog.close()
