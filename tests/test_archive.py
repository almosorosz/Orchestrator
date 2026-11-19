import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import asdict
from contextlib import closing
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialog, QListWidget
from experiment.archive import campaign_details, plot_traces, export_runs
from experiment.models import Campaign, ExperimentDefinition
from experiment.recording import RunRecorder, atomic_json
from experiment.recovery import WriterLease, recover_campaign, recover_roots

APP = QApplication.instance() or QApplication([])


def fixture():
    configs = {"pump": {"name": "Demo pump", "type": "masterflex", "simulated": True, "connection": {}}}
    definitions = tuple(ExperimentDefinition.from_frame(pd.DataFrame({"time_s": [0, 1], "pump": [10, 20]}),
                        configs, name=f"Experiment {i}") for i in range(1, 4))
    return configs, Campaign("Archive demo", definitions)


def wait(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        APP.processEvents()
        time.sleep(.01)
    if not predicate():
        raise AssertionError("Timed out waiting for archive worker")


class ArchiveTests(unittest.TestCase):
    def test_time_range_filters_before_reduction(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            (Path(root) / "execution.csv").write_text(
                "elapsed_s,pump.speed_rpm\n" + "".join(f"{t},{t}\n" for t in range(10000))
            )
            trace = plot_traces({"folder": root}, max_points=20, time_range=(200, 210))
            self.assertEqual(trace["pump.speed_rpm"], [(t, t) for t in range(200, 211)])

    def test_clear_discards_plot_finishing_in_background(self):
        from PySide6.QtGui import QImage
        from gui.campaign_browser import CampaignBrowser

        browser = CampaignBrowser()
        try:
            job = {"kind": "plot", "revision": browser._plot_revision, "runs": {"campaign": ["run"]}}
            browser._pending = job
            browser.clear_plots()
            browser._finished(job, QImage(1, 1, QImage.Format.Format_RGB32), "")
            self.assertIsNone(browser._pending)
            self.assertTrue(browser._image.isNull())
            self.assertEqual(browser.status.text(), "Plot area cleared.")
        finally:
            browser.shutdown()

    def test_selected_experiments_overlay_on_shared_device_panel(self):
        from matplotlib.figure import Figure
        from gui.campaign_browser import render_recorded
        fig = Figure()
        details = {'runs': [{'run_id': str(i), 'index': i + 1} for i in range(5)],
                   'devices': {'pump': {'name': 'Pump', 'type': 'masterflex'}}}
        with patch('matplotlib.figure.Figure', return_value=fig), \
             patch('gui.campaign_browser.plot_traces', return_value={
                 'pump.target': [(0, 0), (1, 10)], 'pump.speed_rpm': [(0, 0), (1, 9)]}):
            result = render_recorded([(details, [str(i) for i in range(5)])], None, True)
        self.assertFalse(result.isNull())
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].lines), 10)
        self.assertEqual(len({line.get_color() for line in fig.axes[0].lines}), 5)

    def test_multiple_campaigns_keep_selections_and_plot_only_checked_devices(self):
        from matplotlib.figure import Figure
        from gui.campaign_browser import CampaignBrowser

        configs, template = fixture()
        configs.update(temperature={"name": "Temperature", "type": "huber"},
                       scale={"name": "Scale", "type": "mettler"})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            root = Path(root)
            for index, name in enumerate(("First", "Second"), 1):
                folder = root / f"campaign_{index}"
                run = folder / "run1"
                run.mkdir(parents=True)
                campaign = {**asdict(template), "name": name, "campaign_id": str(index)}
                atomic_json(folder / "definition.json", {"campaign": campaign, "devices": configs})
                atomic_json(run / "run.json", {"run_id": "run1", "experiment_id": template.experiments[0].experiment_id,
                                               "status": "completed", "recording_complete": True})
                (run / "planned.csv").write_text("time_s,pump\n0,10\n", encoding="utf-8")
                (run / "execution.csv").write_text(
                    f"elapsed_s,pump.speed_rpm,temperature.temp_external,scale.weight_g\n0,{index},25,60\n1,{index + 2},24,59\n",
                    encoding="utf-8")
            browser = CampaignBrowser()
            try:
                browser.root_input.setText(str(root))
                browser.refresh()
                wait(lambda: not browser._busy and len(browser.rows) == 2)
                browser._select_row(browser.campaigns, 1)
                self.assertEqual(browser.runs.rowCount(), 6)
                self.assertEqual(len({browser.runs.item(row, 2).text() for row in (0, 3)}), 2)

                def choose(dialog):
                    choices = dialog.findChild(QListWidget)
                    for index in range(choices.count()):
                        if choices.item(index).text().startswith("scale:"):
                            choices.item(index).setCheckState(Qt.CheckState.Unchecked)
                    return QDialog.DialogCode.Accepted

                with patch.object(QDialog, "exec", choose):
                    browser.device.click()
                self.assertIn("2 of 3 devices", browser.device.text())
                figure = Figure()
                with patch("matplotlib.figure.Figure", return_value=figure):
                    browser.plot_selected()  # Include all executed runs across both campaigns.
                    wait(lambda: not browser._busy and not browser._image.isNull())
                self.assertEqual(len(figure.axes), 2)
                for axis in figure.axes:
                    self.assertEqual(len(axis.lines), 2)
                    self.assertEqual({line.get_label().split()[0] for line in axis.lines}, {"C1", "C2"})
                browser._select_row(browser.runs, 0)
                browser._select_row(browser.runs, 3)
                browser.refresh()
                wait(lambda: not browser._busy)
                self.assertEqual(len(browser.selected_campaigns), 2)
                self.assertEqual(len(browser.selected_runs()), 2)
                self.assertIn("2 of 3 devices", browser.device.text())
                with patch("gui.campaign_browser.QFileDialog.getExistingDirectory", return_value=str(root)):
                    browser.export_selected()
                wait(lambda: not browser._busy)
                exported, = root.glob("experiment_export_*")
                self.assertEqual(len(list(exported.glob("*/run1/execution.csv"))), 2)
            finally:
                self.assertTrue(browser.shutdown())
                browser.close()

    def test_archive_plot_during_actual_simulated_loop_execution(self):
        from MAIN_WINDOW import MainWindow
        from experiment.models import RunStatus
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            experiment_root, model_root = Path(root) / "experiments", Path(root) / "models"
            experiment_root.mkdir()
            model_root.mkdir()
            settings = {"enabled": True, "sql_enabled": False, "output_folder": root, "remote_sql_url": ""}
            with patch("gui.recording_settings_dialog.load_recording_settings", return_value=settings):
                window = MainWindow()
                try:
                    panel = window.control_panel
                    panel.load_simulated_demo()
                    panel.hardware["TEM_MAIN"].connect()
                    panel.device_widgets["TEM_MAIN"]._update_connection_state()
                    window.settings_page.exp_path_input.setText(str(experiment_root))
                    window.settings_page.MODEL_path_input.setText(str(model_root))
                    window.settings_page.exp_number_input.setText("3")
                    self.assertTrue(window.define_command())
                    configurations, _ = panel.collect_current_configuration()
                    definitions = tuple(ExperimentDefinition.from_frame(pd.DataFrame({"time_s": [0, 30 if i == 1 else .3], "TEM_MAIN": [20, 25]}),
                                        configurations, name=f"Live loop {i}") for i in range(3))
                    window._start_loop_campaign(Campaign("Live loop archive", definitions))
                    dashboard = window.experiment_page.dashboard
                    wait(lambda: dashboard.worker is not None and dashboard.worker.runs[0].status == RunStatus.COMPLETED
                         and dashboard.worker.runs[1].status == RunStatus.RUNNING)
                    from experiment.archive import read_json
                    first_run = dashboard.worker.runs[0].run_id
                    wait(lambda: read_json(dashboard.worker.recorder.folder / first_run / "run.json", {}).get("recording_complete"))
                    browser = window.plots_page.browser
                    browser.root_input.setText(str(experiment_root))
                    browser.refresh()
                    wait(lambda: not browser._busy and browser.selected_campaigns and browser.selected_campaigns[0]["completed"] == 1)
                    self.assertEqual([r["status"] for _, r in browser.run_rows], ["completed", "running", "queued"])
                    browser.runs.selectRow(0)
                    browser.plot_selected()
                    wait(lambda: not browser._busy and not browser._image.isNull())
                    self.assertTrue(dashboard.is_execution_active())
                    self.assertTrue(window.execution.timer.isActive())
                finally:
                    window.stop_command()
                    dashboard = window.experiment_page.dashboard
                    if dashboard is not None:
                        dashboard.stop_orchestrator()
                        wait(lambda: dashboard.thread is None and not dashboard.is_execution_active())
                        window.end_experiment_session()
                    window.close()
                    APP.processEvents()

    def test_completed_plots_while_other_runs_are_running_and_queued(self):
        from gui.campaign_browser import CampaignBrowser
        configs, campaign = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            recorder = RunRecorder(root, campaign, configs)
            browser = CampaignBrowser()
            browser.root_input.setText(root)
            try:
                for i in range(2):
                    run_id = f"run{i}"
                    recorder.submit("run_start", run_id, {"experiment_id": campaign.experiments[i].experiment_id,
                        "planned": {"time_s": [0, 1], "pump": [10, 20]}})
                    for second in range(3):
                        recorder.submit("sample", run_id, {"elapsed_s": second, "pump.target": 10 + second, "pump.speed_rpm": second})
                    if i == 0:
                        recorder.submit("run_end", run_id, {"run_id": run_id, "experiment_id": campaign.experiments[i].experiment_id, "status": "completed"})
                wait(lambda: (recorder.folder / "run1" / "execution.csv").exists())
                browser.set_roots([root])
                wait(lambda: not browser._busy and browser.selected_campaigns)
                self.assertEqual([r["status"] for _, r in browser.run_rows], ["completed", "running", "queued"])
                browser.runs.selectRow(0)
                ticks = []
                timer = QTimer()
                timer.setInterval(10)
                timer.timeout.connect(lambda: ticks.append(True))
                timer.start()
                browser.plot_selected()
                wait(lambda: not browser._busy and not browser._image.isNull())
                timer.stop()
                self.assertGreater(len(ticks), 2)
                self.assertIsNone(recorder.error)
                self.assertTrue(recorder.thread.is_alive())
                details = campaign_details(recorder.folder)
                trace = plot_traces(details["runs"][0])
                self.assertEqual([v[0] for v in trace["pump.speed_rpm"]], [0, 1, 2])
                output = Path(export_runs(details, ["run0"], root))
                self.assertTrue((output / "run0" / "execution.csv").exists())
                exported = json.loads((output / "definition.json").read_text())
                self.assertNotIn("settings", exported)
                self.assertNotIn("connection", exported["devices"]["pump"])
                with self.assertRaises(ValueError):
                    export_runs(details, ["run1"], root)
                self.assertEqual(recover_campaign(recorder.folder)["status"], "active")
            finally:
                recorder.close()
                self.assertTrue(browser.shutdown())
                browser.close()

    def test_snapshot_ignores_partial_csv_tail_and_bounds_plot_memory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            path = Path(root) / "execution.csv"
            with path.open("w", newline="") as stream:
                stream.write("elapsed_s,pump.speed_rpm\n")
                for i in range(30000):
                    stream.write(f"{i},{10000 if i == 15001 else i % 50}\n")
                stream.write("30000,123")
            trace = plot_traces({"folder": root})["pump.speed_rpm"]
            self.assertLessEqual(len(trace), 4000)
            self.assertEqual(trace[-1][0], 29999)
            self.assertEqual(max(v[1] for v in trace), 10000)

    def test_recovery_is_idempotent_and_marks_interruption(self):
        configs, campaign = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            folder = Path(root) / "campaign_crash"
            run = folder / "run1"
            run.mkdir(parents=True)
            atomic_json(folder / "definition.json", {"campaign": asdict(campaign), "devices": configs, "settings": {"sql_enabled": False}})
            event = {"id": "start1", "utc": "2026-11-04T00:00:00Z", "kind": "run_start", "run_id": "run1",
                     "payload": {"experiment_id": campaign.experiments[0].experiment_id}}
            (folder / "events.jsonl").write_text(json.dumps(event) + "\n")
            pending = run / "samples.pending.jsonl"
            pending.write_text(''.join(json.dumps({"elapsed_s": i, "pump.speed_rpm": i}) + "\n" for i in range(4)) + '{"partial":')
            (run / "planned.csv").write_text("time_s,pump\n0,10\n")
            lease = WriterLease(folder)
            self.assertTrue(lease.acquire())
            lease.release()
            self.assertEqual(recover_campaign(folder)["status"], "recovered")
            self.assertEqual(pd.read_csv(run / "execution.csv")["elapsed_s"].tolist(), [0, 1, 2, 3])
            self.assertTrue((run / "samples.pending.jsonl.partial").exists())
            self.assertEqual(json.loads((run / "run.json").read_text())["status"], "interrupted")
            self.assertEqual(campaign_details(folder)["status"], "interrupted")
            self.assertEqual(recover_campaign(folder)["status"], "unchanged")
            self.assertEqual(len(pd.read_csv(run / "execution.csv")), 4)

    def test_recovery_lock_and_csv_lock_do_not_damage_data(self):
        configs, campaign = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            folder = Path(root) / "campaign_crash"
            directory = folder / "run1"
            directory.mkdir(parents=True)
            atomic_json(folder / "definition.json", {"campaign": asdict(campaign), "devices": configs})
            pending = directory / "samples.pending.jsonl"
            pending.write_text('{"elapsed_s":0,"pump.speed_rpm":10}\n')
            lease = WriterLease(folder)
            self.assertTrue(lease.acquire())
            try:
                self.assertEqual(recover_campaign(folder)["status"], "active")
                self.assertTrue(pending.exists())
            finally:
                lease.release()
            original = Path.open
            def opening(path, *args, **kwargs):
                if path.name == "execution.csv":
                    raise PermissionError("Excel lock")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", opening):
                self.assertEqual(recover_roots([root])[0]["status"], "waiting")
            self.assertTrue(pending.exists())
            self.assertEqual(recover_campaign(folder)["status"], "recovered")
            self.assertEqual(len(pd.read_csv(directory / "execution.csv")), 1)

    def test_legacy_recovery_requires_explicit_selection(self):
        configs, campaign = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            folder = Path(root) / "campaign_legacy"
            folder.mkdir()
            atomic_json(folder / "definition.json", {"campaign": asdict(campaign), "devices": configs})
            self.assertEqual(recover_campaign(folder)["status"], "waiting")
            self.assertFalse((folder / ".writer.lock").exists())
            self.assertEqual(recover_campaign(folder, allow_legacy=True)["status"], "recovered")

    def test_sql_recovery_preserves_ids_and_does_not_duplicate_events(self):
        import sqlite3
        configs, campaign = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            folder = Path(root) / "campaign_sql"
            directory = folder / "run1"
            directory.mkdir(parents=True)
            lease = WriterLease(folder)
            lease.acquire()
            lease.release()
            atomic_json(folder / "definition.json", {"campaign": asdict(campaign), "devices": configs, "settings": {"sql_enabled": True}})
            events = [{"id": "start", "utc": "now", "kind": "run_start", "run_id": "run1", "payload": {"experiment_id": campaign.experiments[0].experiment_id}},
                      {"id": "sample", "utc": "now", "kind": "sample", "run_id": "run1", "payload": {"elapsed_s": 0, "pump.speed_rpm": 10}}]
            (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            (directory / "samples.pending.jsonl").write_text(json.dumps(events[1]["payload"]) + "\n")
            self.assertEqual(recover_campaign(folder)["status"], "recovered")
            self.assertEqual(recover_campaign(folder)["status"], "unchanged")
            with closing(sqlite3.connect(Path(root) / "recordings.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT id FROM events ORDER BY id").fetchall(), [("sample",), ("start",)])


class PresetTests(unittest.TestCase):
    def test_setup_round_trip_does_not_connect_or_reuse_generated_campaign(self):
        from MAIN_WINDOW import MainWindow
        from experiment.doe_campaign import ProfileAssignment
        from app.presets import read_preset
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            settings = {"enabled": True, "sql_enabled": False, "output_folder": root, "remote_sql_url": ""}
            with patch("gui.recording_settings_dialog.load_recording_settings", return_value=settings):
                window = MainWindow()
                try:
                    panel = window.control_panel
                    panel.load_simulated_demo()
                    panel.control_output_path.setText(str(Path(root) / "manual"))
                    panel.device_widgets["STR_MAIN"].config["target_limits"] = {"min": 0, "max": 900, "min_nonzero": 0}
                    window._doe_assignments = (ProfileAssignment("T", "TEM_MAIN"),)
                    window.experiment_service.exp_par.T_room = 22
                    window.settings_page.exp_path_input.setText(str(Path(root) / "loop"))
                    filename = str(Path(root) / "setup.json")
                    with patch("MAIN_WINDOW.QFileDialog.getSaveFileName", return_value=(filename, "")):
                        window.save_setup_preset()
                    data, parameters, assignments = read_preset(filename)
                    self.assertEqual(parameters.T_room, 22)
                    self.assertEqual(assignments[0].device_id, "TEM_MAIN")
                    window.experiment_service.exp_par.T_room = 10
                    window.create_experiments = object()
                    window._assigned_doe_campaign = object()
                    with patch("MAIN_WINDOW.QFileDialog.getOpenFileName", return_value=(filename, "")), patch("gui.recording_settings_dialog.save_recording_settings") as save:
                        window.load_setup_preset()
                        save.assert_called_once()
                    self.assertEqual(window.experiment_service.exp_par.T_room, 22)
                    self.assertIsNone(window.create_experiments)
                    self.assertIsNone(window._assigned_doe_campaign)
                    # Settings owns the shared output root for every start path.
                    self.assertEqual(window.settings_page.exp_path_input.text(), str(Path(root) / "loop"))
                    self.assertEqual(panel.control_output_path.text(), str(Path(root) / "loop"))
                    self.assertEqual(panel.device_widgets["STR_MAIN"].config["target_limits"]["max"], 900)
                    self.assertFalse(any(device.is_connected for device in panel.hardware.values()))
                    self.assertFalse(window.execution.timer.isActive())
                    data["version"] = 999
                    Path(filename).write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        read_preset(filename)
                finally:
                    window.close()
                    APP.processEvents()
