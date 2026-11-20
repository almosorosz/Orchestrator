"""Run phases survive files and delimit the observations used by BO and IMED."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from app.presets import read_preset, write_preset
from bo.profiles import Profile, case_runs, command_profiles, make_profile
from experiment.adaptive_loop import RecipeCompiler, default_setup, read_completed_recordings
from experiment.doe_campaign import DoeCampaignService, ProfileAssignment
from experiment.doe_snapshot import restore_doe, snapshot_doe
from experiment.models import Campaign, ControlModePoint, DeviceProfile, ExperimentDefinition, ProfilePoint
from experiment.run_stages import PURPOSES, RunStage, frame_metadata, modeling_campaign, stage_window
from experiment_dashboard.experiment_dashboard import ExperimentWorker
from gui.external_model_info import experiment_template
from gui.loop_setup_dialog import LoopSetupDialog
from hardware_devices.simulated import SimulatedDevice
from test_recipe_editor import APP, CONFIGS
import test_recipe_editor as recipe_fixture


def phases():
    labels = (("prep", "Prepare vessel", "Preparing for experiment", 0., 2.),
              ("dissolve", "Dissolution", "Dissolution", 2., 4.),
              ("seed", "Seeding", "Seeding", 4., 5.),
              ("exp", "Crystallize", "Experiment", 5., 8.),
              ("eq", "Equilibrate", "Equilibration", 8., 10.),
              ("clean", "Clean vessel", "Cleaning", 10., 12.))
    return tuple(RunStage(*values) for values in labels)


def manual_campaign():
    frame = pd.DataFrame({"time_s": [0., 2., 4., 5., 8., 10., 12.],
                          "TEMP": [25., 25., 25., 25., 24., 24., 40.]})
    frame.attrs["run_stages"] = [asdict(stage) for stage in phases()]
    frame.attrs["model_window"] = [5., 10.]
    return Campaign("Manual phases", (ExperimentDefinition.from_frame(frame, CONFIGS),))


class RunStageTests(unittest.TestCase):
    def test_dashboard_stage_selection_is_applied_to_starting_and_followup_batches(self):
        from test_adaptive_loop import AdaptiveLoopTests, PRESET
        from imed_python.demo_campaign import model_inputs

        fixture = AdaptiveLoopTests()
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window, _, models, logs = fixture.setup_dashboard(Path(directory))
            try:
                frames = []
                for temperature in (14., 19., 24.):
                    frame = pd.DataFrame({"time_s": [0., 5., 10., 12.], "T1": [25., temperature, temperature, 40.]})
                    frame.attrs["run_stages"] = [asdict(stage) for stage in phases()]
                    frames.append(frame)
                window.control_panel.experiments = frames
                _, _, process, solubility, _ = model_inputs(False)
                window.imed_page.panel.restore_setup_settings({"Solubility": solubility})
                setup = window.settings_page.loop_setup
                setup["manual_assignments"] = {"T": "T1"}
                setup["process"] = process
                setup["recording"].update(no_feed=True, initial_concentration=.1)
                setup["channels"] = {"Concentration": ""}
                setup["experiment_period"] = {"mode": "stages", "stage_ids": ["exp", "eq"]}
                bo = json.loads(PRESET.with_name("bo_validation_loop.json").read_text())
                bo["targets"]["d50"]["enabled"] = False
                bo["targets"]["yield"]["enabled"] = True
                bo["physical_profiles"] = {}
                window.bo_page.restore(bo)
                with patch.object(window, "_choose_loop_source", return_value="manual"):
                    window.start_command()
                self.assertIsNotNone(window._adaptive_loop, logs)
                self.assertTrue(fixture.wait_for(lambda: window._adaptive_loop is None), logs)
                self.assertFalse(any(level == "ERROR" for level, _ in logs), logs)
                self.assertTrue(any("modeling period for Experiment 1: 5–10 s" in message for _, message in logs), logs)
                self.assertTrue(all(run["modeling_window_s"][0] == 5. for run in window.bo_page.state["runs"]))
                recipe_file, = models.glob("*/recommendation.json")
                experiment = json.loads(recipe_file.read_text())["campaign"]["experiments"][0]
                self.assertEqual(experiment["model_window"][0], 5.)
                self.assertEqual(experiment["run_stages"][0]["purpose"], "Preparing for experiment")
                self.assertEqual(experiment["run_stages"][-1]["purpose"], "Cleaning")
                self.assertEqual(len(window.bo_page.state["runs"]), 4)
            finally:
                fixture.doCleanups()

    def test_sampled_stage_durations_survive_exact_restore_and_resolve_per_run(self):
        self.assertTrue({"Preparing for experiment", "Dissolution", "Seeding"} <= set(PURPOSES))
        service = recipe_fixture.RecipeEditorTests().service()
        from experiment.recipe import parameter
        service.exp_par.timeline["stages"][0]["duration"] = parameter(2., "Alternate", 2., 4.)
        creator = service.generate_initial_doe(2, "LHS")
        saved = snapshot_doe(creator)
        # Load older exact snapshots, which have durations but no phase metadata.
        for snapshot in saved["parameter_snapshots"].values():
            snapshot.pop("run_stages")
        restored = restore_doe(service, saved)
        roles = tuple(ProfileAssignment(lane["id"], lane["device_id"]) for lane in service.exp_par.timeline["lanes"])
        campaign = DoeCampaignService(restored).compile(roles, CONFIGS)
        setup = default_setup()
        setup["experiment_period"] = {"mode": "stages", "stage_ids": ["cryst", "eq"]}
        resolved = modeling_campaign(campaign, setup)
        self.assertEqual([exp.model_window for exp in resolved.experiments], [(2., 9.), (4., 11.)])
        self.assertEqual(resolved.experiments[1].run_stages[1].start_s, 4.)
        self.assertEqual(resolved.campaign_id, campaign.campaign_id)

    def test_period_rejects_gaps_missing_ids_and_out_of_range_manual_times(self):
        with self.assertRaisesRegex(ValueError, "continuous"):
            stage_window(phases(), ["exp", "clean"])
        with self.assertRaisesRegex(ValueError, "missing"):
            stage_window(phases(), ["deleted-stage"])
        setup = default_setup()
        setup["experiment_period"] = {"mode": "manual", "stage_ids": []}
        setup["recording"].update(start_s=5., end_s=20.)
        with self.assertRaisesRegex(ValueError, "inside"):
            modeling_campaign(manual_campaign(), setup)

    def test_manual_phases_preserve_prep_cleanup_in_both_followup_backends(self):
        campaign = manual_campaign()
        setup = default_setup()
        setup["experiment_period"] = {"mode": "stages", "stage_ids": ["exp", "eq"]}
        for backend in ("bo", "imed"):
            compiler = RecipeCompiler(setup, campaign, CONFIGS, None, (ProfileAssignment("T", "TEMP"),), (), backend)
            result, window = compiler.build({"temperature": Profile(np.array([0., 7.]), np.array([25., 20.]), "Proposed")},
                                            {"SeedMass": 0.}, name="Follow-up", source=backend, default_seed=0.)
            self.assertEqual(window, (5., 12.))
            experiment = result.experiments[0]
            self.assertEqual([stage.purpose for stage in experiment.run_stages],
                             ["Preparing for experiment", "Dissolution", "Seeding", "Experiment", "Equilibration", "Cleaning"])
            self.assertEqual(experiment.run_stages[-1].start_s, 12.)
            frame = experiment.to_frame().set_index("time_s")
            self.assertEqual(frame.loc[2., "TEMP"], 25.)
            self.assertEqual(frame.loc[np.nextafter(12., 5.), "TEMP"], 20.)
            self.assertEqual(frame.loc[12., "TEMP"], 24.)
            self.assertEqual(frame.loc[14., "TEMP"], 40.)

    def test_followup_changes_only_crystallization_with_a_shorter_recommendation(self):
        from dataclasses import replace

        original = manual_campaign().experiments[0]
        temperature = DeviceProfile("TEMP", tuple(ProfilePoint(t, v) for t, v in
            ((0., 20.), (2., 65.), (4., 65.), (5., 40.), (8., 25.), (10., 25.), (12., 45.))), unit="°C")
        original = replace(original, profiles=(temperature,), seed_mass_g=.2,
                           control_modes=(ControlModePoint(0., "TEMP", "Jacket"),
                                          ControlModePoint(5., "TEMP", "Process"),
                                          ControlModePoint(8., "TEMP", "Jacket")))
        setup = default_setup()
        setup["experiment_period"] = {"mode": "stages", "stage_ids": ["exp"]}
        setup["followup_start_s"] = 99.
        recommendation = {"temperature": Profile(np.array([0., 2.]), np.array([37., 21.]), "Proposed")}
        for backend in ("bo", "imed"):
            with self.subTest(backend=backend):
                compiler = RecipeCompiler(setup, Campaign("Staged recipe", (original,)), CONFIGS, None,
                                          (ProfileAssignment("T", "TEMP"),), (), backend)
                followup, window = compiler.build(recommendation, {"SeedMass": .2}, name="Follow-up",
                                                 source=backend, default_seed=.2)
                experiment = followup.experiments[0]
                self.assertEqual(window, (5., 7.))
                self.assertEqual(experiment.run_stages[:3], original.run_stages[:3])
                self.assertEqual([s.id for s in experiment.run_stages], [s.id for s in original.run_stages])
                self.assertEqual((experiment.run_stages[4].start_s, experiment.run_stages[4].end_s), (7., 9.))
                self.assertEqual(experiment.control_modes[-1], ControlModePoint(7., "TEMP", "Jacket"))
                frame = experiment.to_frame()
                time, values = frame["time_s"], frame["TEMP"]
                np.testing.assert_allclose(np.interp([2., 4., 4.5], time, values), [65., 65., 52.5])
                np.testing.assert_allclose(np.interp([5., 6., 7., 8., 9., 10., 11.], time, values),
                                           [37., 29., 25., 25., 25., 35., 45.])
                commands = command_profiles(asdict(experiment), {"SeedMass": .2}, {"devices": CONFIGS},
                                            5., 2., {"T": "TEMP"})
                self.assertEqual(commands["temperature"].values[-1], 21.)
                with self.assertRaisesRegex(ValueError, "changes the seed dose"):
                    compiler.build(recommendation, {"SeedMass": .3}, name="Follow-up", source=backend, default_seed=.2)

    def test_complete_recording_is_retained_but_model_and_bo_use_selected_stages(self):
        campaign = modeling_campaign(manual_campaign(), default_setup())
        definition = campaign.experiments[0]
        configs = {"TEMP": deepcopy(CONFIGS["TEMP"])}
        device = SimulatedDevice("TEMP", "huber", noise=0.)
        device.connect()
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            worker = ExperimentWorker({"TEMP": device}, configs, list(campaign.experiments), speed=100.,
                                      end_hold_s=0., campaign=campaign,
                                      recording={"enabled": True, "sql_enabled": False, "output_folder": directory,
                                                 "remote_sql_url": ""})
            worker.start_experiment()
            self.assertEqual(worker.outcome.value, "completed", worker.error_message)
            self.assertTrue(worker.recorder.complete.wait(10))
            folder = Path(worker.recorder.folder)
            recording = default_setup()["recording"]
            recording.update(no_feed=True, initial_concentration=.1)
            measured, processes, runs = read_completed_recordings(folder, (definition.experiment_id,), recording,
                {"Temperature": "TEMP.temp_external"}, {definition.experiment_id: {"SeedMass": 0., "SolvMass": 75., "AntiSolvMass": 0.}},
                {definition.experiment_id: definition.model_window})
            full = pd.read_csv(runs[0][2])
            self.assertEqual(full.elapsed_s.max(), 12.)
            self.assertEqual(measured[0]["RecordingStart_s"], 5.)
            np.testing.assert_allclose(measured[0]["TemperatureData"][0, [0, -1]], [0., 5.])
            self.assertEqual([s["id"] for s in measured[0]["ModelingStages"]], ["exp", "eq"])
            bo = case_runs({"ExpData": measured, "ProcInf": processes}, folder, controls={"T": "TEMP"})[0]
            self.assertEqual(bo["metrics"]["duration"], 5.)
            self.assertEqual(bo["modeling_window_s"], [5., 10.])
            np.testing.assert_allclose(make_profile(bo["profiles"]["temperature"]["points"], "Test").values, [25., 24., 24.])
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
            stages = [event["payload"] for event in events if event["kind"] == "run_stage"]
            self.assertEqual([s["id"] for s in stages if s["modeling_period"]], ["exp", "eq"])
            self.assertEqual(len(stages), 6)
        device.disconnect()

    def test_control_panel_generation_template_and_presets_preserve_phase_metadata(self):
        from MAIN_WINDOW import MainWindow
        window = MainWindow()
        try:
            panel = window.control_panel
            configs = {"TEMP": deepcopy(CONFIGS["TEMP"])}
            panel.apply_loaded_configuration(configs, {}, [])
            panel.device_widgets["TEMP"].profile = pd.DataFrame({"time_s": [0., 12.], "value": [25., 24.]})
            panel.manual_run_stages = [asdict(stage) for stage in phases()]
            panel.generate_experiment()
            original = panel.experiments[0]
            self.assertEqual(original.attrs["model_window"], (5., 10.))
            with TemporaryDirectory(dir=Path.cwd()) as directory:
                path = Path(directory) / "external.json"
                payload = experiment_template({"devices": configs, "experiments": [original.to_dict(orient="split")],
                                                "experiment_metadata": [frame_metadata(original)]})
                path.write_text(json.dumps(payload))
                with patch("control_panel.control_panel.QFileDialog.getOpenFileName", return_value=(str(path), "")), \
                     patch("control_panel.control_panel.QMessageBox.information"), \
                     patch("control_panel.control_panel.QMessageBox.critical") as error:
                    panel.load_configuration()
                error.assert_not_called()
                self.assertEqual(ExperimentDefinition.from_frame(panel.experiments[0], configs).run_stages, phases())
                preset = Path(directory) / "setup.json"
                write_preset(preset, payload, window.experiment_service.exp_par, (), {},
                             {"enabled": True, "sql_enabled": False, "output_folder": directory, "remote_sql_url": ""})
                saved, _, _ = read_preset(preset)
                self.assertEqual(saved["control_panel"]["experiment_metadata"][0]["model_window"], [5., 10.])
        finally:
            window.close()
            APP.processEvents()

    def test_loop_stage_selector_persists_and_rejects_nonadjacent_selection(self):
        campaign = manual_campaign()
        setup = default_setup()
        setup["experiment_period"] = {"mode": "stages", "stage_ids": ["exp", "eq"]}
        for backend in ("bo", "imed"):
            dialog = LoopSetupDialog(setup, backend)
            try:
                dialog.period.set_context({"campaign": campaign, "run_stages": phases()})
                self.assertEqual(dialog.collect()["experiment_period"], setup["experiment_period"])
                dialog.period.table.item(4, 0).setCheckState(Qt.CheckState.Unchecked)
                dialog.period.table.item(5, 0).setCheckState(Qt.CheckState.Checked)
                dialog.accept()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
                self.assertIn("continuous", dialog.error.text())
                dialog.period.table.item(4, 0).setCheckState(Qt.CheckState.Checked)
                dialog.accept()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted, dialog.error.text())
                self.assertEqual(dialog.setup["experiment_period"]["stage_ids"], ["exp", "eq", "clean"])
            finally:
                dialog.close()
