"""Shared physical assignments, scalar mixing, stocks and executable BO controls."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from experiment.adaptive_loop import default_setup, read_completed_recordings, validate_setup, RecipeCompiler
from experiment.doe_campaign import ProfileAssignment
from experiment.loop_assignments import resolve_loop_setup
from experiment.material_balance import initial_charges
from experiment.models import Campaign, DeviceProfile, ExperimentDefinition, ProfilePoint
from experiment.pump_calibration_files import seed_delivery_rate
from experiment.reservoirs import ReservoirBook, InsufficientStock
from imed_python.campaign_import import seed_distribution
from imed_python.mixing import average_mixing
from bo.physical import Stage, physical_profile, physical_controls, apply_controls, control_values
from bo.profiles import Profile
from test_recipe_editor import APP


def fixture(stock=15.):
    topology = {"version": 1, "modules": [{"code": "M", "name": "Lab"}], "unit_operations": [
        {"code": "feed", "name": "Solvent tank", "kind": "Container", "module": "M", "material_role": "solvent",
         "inventory_g": stock, "inventory_id": "test-refill", "density_g_ml": 1.},
        {"code": "C", "name": "Crystallizer", "kind": "Reactor", "module": "M", "material_role": "reactor"}]}
    configs = {key: {"name": key, "type": kind, "simulated": True, "connectivity": p} for key, kind, p in (
        ("T", "huber", {"role": "controlled", "module": "M", "unit": "C"}),
        ("S", "jkem", {"role": "controlled", "module": "M", "unit": "C"}),
        ("P", "ismatec", {"role": "controlled", "module": "M", "source": "feed", "target": "C"}),
        ("V", "nidaq_valve", {"role": "controlled", "module": "M", "unit": "C"}))}
    return topology, configs


def profile(device, values, step=False):
    return DeviceProfile(device, tuple(ProfilePoint(float(t), float(v)) for t, v in values), "step" if step else "linear")


def process():
    return {"SolvMass": 100., "AntiSolvMass": 0., "SolventDensity": 1., "AntiSolventDensity": 2.,
            "SolidDensity": 1.2, "ShapeFactor": .52, "SeedMass": 0., "PowerRatio": 1.}


class LinkedInputTests(unittest.TestCase):
    def test_topology_resolves_roles_and_measurements_without_duplicate_map(self):
        topology, configs = fixture()
        configs["D"] = {"name": "Size probe", "type": "opcua", "simulated": True,
                         "connectivity": {"role": "read_only", "module": "M", "unit": "C", "quantity": "D50"}}
        setup, report = resolve_loop_setup(default_setup(), topology, configs, strict=True)
        self.assertEqual(setup["manual_assignments"], {"T": "T", "stirring_rate": "S", "solvent": "P", "valve_state": "V"})
        self.assertEqual(setup["channels"]["D50"], "D.value")
        self.assertEqual(report["origins"]["solvent"], "MSI topology")

    def test_ambiguous_pumps_need_a_choice_and_wrong_direction_is_rejected(self):
        topology, configs = fixture()
        configs["P2"] = deepcopy(configs["P"])
        with self.assertRaisesRegex(ValueError, "several candidates"):
            resolve_loop_setup(default_setup(), topology, configs, strict=True)
        setup = default_setup()
        setup["manual_assignments"] = {"solvent": "P"}
        resolve_loop_setup(setup, topology, configs, strict=True)
        configs["P"]["connectivity"].update(source="C", target="feed")
        with self.assertRaisesRegex(ValueError, "direction"):
            resolve_loop_setup(setup, topology, configs, strict=True)

    def test_matching_applied_doe_assignment_is_reused(self):
        _, configs = fixture()
        campaign = Campaign("Manual", (ExperimentDefinition("Run", (profile("T", [(0, 25), (10, 25)]),)),))
        setup, report = resolve_loop_setup(default_setup(), {"unit_operations": []}, configs, campaign=campaign,
                                           fallback={"T": "T", "AS": "unused"}, strict=True)
        self.assertEqual(setup["manual_assignments"], {"T": "T"})
        self.assertEqual(report["origins"]["T"], "Reused DoE assignment")

    def test_seed_delivery_reuses_loaded_mass_calibration(self):
        config = {"type": "masterflex", "pump_calibration": {"version": 1, "input_unit": "g/h", "output_unit": "RPM",
                   "points": [[0., 0.], [72., 100.]]}}
        self.assertEqual(seed_delivery_rate(config, {"rpm": 50., "use_device_calibration": True}), .01)
        setup = default_setup()
        setup["seed_pump"] = {"rpm": 50., "grams_per_second": None, "use_device_calibration": True}
        validate_setup(setup)

    def test_unseeded_zero_percentiles_have_no_seed_population(self):
        from imed_python.distribution import seed_number_density
        self.assertEqual(seed_distribution([0, 0, 0]).shape, (2, 0))
        setup = default_setup()
        setup["recording"]["seed_percentiles"] = [0., 0., 0.]
        validate_setup(setup)
        np.testing.assert_array_equal(seed_number_density({}, {"SeedMass": 0.}, np.linspace(0, 100, 11)), np.zeros(10))

    def test_scalar_power_uses_mean_rpm_and_mean_volume_and_off_is_one(self):
        experiment = {"TemperatureData": np.array([[0., 10.], [25., 20.]]),
                      "FlowRateData": np.array([[0., 10.], [999., 999.]]),
                      "AddedAntisolventData": np.array([[0., 10.], [0., 10.]])}
        settings = {"enabled": True, "power_number": 2., "diameter_m": .02, "volume_mode": "charged", "volume_ml": 100.}
        result = average_mixing(experiment, process(), settings, mean_rpm=200.)
        expected = 2 * (200 / 60)**3 * .02**5 / (102.5e-6)
        self.assertAlmostEqual(result["PowerRatio"], expected)
        self.assertEqual(result["MixingSummary"]["average_volume_ml"], 102.5)
        self.assertNotIn("StirringRateData", experiment)
        self.assertNotIn("VolumeData", experiment)
        self.assertEqual(average_mixing(experiment, process(), {**settings, "enabled": False})["PowerRatio"], 1.)

    def test_initial_charge_is_derived_and_explicit_override_wins(self):
        _, configs = fixture()
        experiment = ExperimentDefinition("Charge", (profile("T", [(0, 25), (15, 25)]),
            profile("P", [(0, 60), (5, 0), (15, 0)], True)), model_window=(5, 15))
        result = initial_charges(experiment, process(), {"solvent": "P"}, configs)
        self.assertEqual(result["SolvMass"], 5.)
        result = initial_charges(experiment, process(), {"solvent": "P"}, configs, overrides={"SolvMass": 100.})
        self.assertEqual(result["SolvMass"], 100.)

    def test_both_backends_retain_volume_scaled_preparation(self):
        _, configs = fixture()
        experiment = ExperimentDefinition("Charge", (profile("T", [(0, 25), (15, 25)]),
            profile("P", [(0, 60), (5, 0), (15, 0)], True)), model_window=(5, 15), initial_mass_g=0.)
        for mode in ("bo", "imed"):
            compiler = RecipeCompiler(default_setup(), Campaign("DoE", (experiment,)), configs, None,
                                      (ProfileAssignment("T", "T"), ProfileAssignment("solvent", "P")), (), mode)
            inputs = {**process(), "SolvMass": 5., "InitialVolume_ml": 5.}
            result, _ = compiler.build({"temperature": Profile(np.array([0., 10.]), np.array([25., 20.]), "Recipe"),
                                       "volume": Profile(np.array([0.]), np.array([10.]), "Charge volume")},
                                      inputs, name="Follow-up", source=mode, default_seed=0.)
            pump = next(p for p in result.experiments[0].profiles if p.device_id == "P")
            self.assertEqual(pump.points[0].target, 120.)
            self.assertEqual(inputs["SolvMass"], 10.)

    def test_borrowed_doe_effects_gate_sensitivity_and_power_parameters(self):
        from experiment.campaign_effects import initial_input_groups, power_effects
        values = [{"StirringRPM": 100., "InitialVolume_ml": 100., "PowerRatio": 1.},
                  {"StirringRPM": 200., "InitialVolume_ml": 200., "PowerRatio": 4.}]
        self.assertEqual(initial_input_groups(values), ["stirring", "volume"])
        table = np.zeros((35, 4))
        table[11, 2:] = [0., 2.]
        models = [{"paraset": table}]
        power_effects(models, values)
        self.assertEqual(models[0]["paraset"][11, 0], 1.)
        power_effects(models, values[:1])
        self.assertEqual(models[0]["paraset"][11, 0], 0.)

    def test_bo_stirring_editor_and_decoding_keep_final_rpm(self):
        from bo.stage_editor import StageEditor
        editor = StageEditor()
        self.assertIn("stirring", editor.editors)
        stages = (Stage(5., 100., 100.), Stage(5., 200., 200.))
        controls = physical_controls("stirring", stages)
        values = control_values(stages, controls)
        values[1] = 150.
        changed = apply_controls("stirring", stages, controls, values)
        self.assertEqual(changed[0].end, 150.)
        result = physical_profile("stirring", changed, 15.)
        np.testing.assert_allclose(result.sample([0., 6., 15.]), [150., 200., 200.])
        editor.close()

    def test_stock_deduction_is_idempotent_and_next_experiment_is_blocked(self):
        topology, configs = fixture()
        experiment = ExperimentDefinition("Dose", (profile("P", [(0, 60), (10, 0)], True),), initial_mass_g=20.)
        book = ReservoirBook()
        book.configure(topology, configs)
        book.begin(experiment)
        self.assertEqual(book.advance(experiment, 0., {}), 20.)
        self.assertEqual(book.advance(experiment, 5., {}), 25.)
        book.advance(experiment, 5., {})
        self.assertEqual(book.remaining["feed"], 10.)
        book.advance(experiment, 10., {})
        self.assertEqual(book.remaining["feed"], 5.)
        with self.assertRaisesRegex(InsufficientStock, "next experiment requires 10"):
            book.check(experiment)

    def test_stock_check_catches_peak_draw_before_a_pump_return(self):
        topology, configs = fixture(stock=2.)
        experiment = ExperimentDefinition("Return", (profile("P", [(0, 60), (10, -60)]),))
        book = ReservoirBook()
        book.configure(topology, configs)
        with self.assertRaisesRegex(InsufficientStock, "requires 2.5"):
            book.check(experiment)

    def test_source_scale_overrides_calculated_draw_and_valve_resets_mass(self):
        topology, configs = fixture()
        configs["W"] = {"type": "mettler", "connectivity": {"unit": "feed"}}
        experiment = ExperimentDefinition("Dose", (profile("P", [(0, 60), (10, 0)], True),), initial_mass_g=20.)
        book = ReservoirBook()
        book.configure(topology, configs)
        book.begin(experiment)
        book.advance(experiment, 0., {"W": {"weight_g": 100.}})
        self.assertEqual(book.advance(experiment, 5., {"W": {"weight_g": 96.}}), 24.)
        self.assertEqual(book.remaining["feed"], 11.)
        self.assertEqual(book.advance(experiment, 5., {"W": {"weight_g": 96.}, "V": {"target_state": 1}}), 0.)
        self.assertEqual(book.advance(experiment, 10., {"W": {"weight_g": 90.}, "V": {"target_state": 0}}), 6.)

    def test_inventory_survives_reload_without_resetting_stock(self):
        topology, configs = fixture()
        experiment = ExperimentDefinition("Dose", (profile("P", [(0, 60), (10, 0)], True),))
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            filename = Path(directory) / "stock.json"
            book = ReservoirBook(state_file=filename)
            book.configure(topology, configs)
            book.begin(experiment)
            book.advance(experiment, 5., {})
            book.save()
            restored = ReservoirBook(state_file=filename)
            restored.configure(topology, configs)
            self.assertEqual(restored.remaining["feed"], 10.)

    def test_recorded_mean_rpm_and_pump_fallback_generate_scalar_inputs(self):
        from experiment.recording import atomic_json
        cfg = {"T": {"type": "huber"}, "S": {"type": "jkem"}, "AS": {"type": "ismatec"}}
        definition = ExperimentDefinition("Measured", (profile("T", [(0, 25), (10, 20)]),))
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            folder = Path(directory)
            atomic_json(folder / "definition.json", {"campaign": asdict(Campaign("Test", (definition,))), "devices": cfg})
            run = folder / "run1"
            run.mkdir()
            atomic_json(run / "run.json", {"experiment_id": definition.experiment_id, "experiment_number": 1,
                                          "status": "completed", "recording_complete": True})
            pd.DataFrame({"elapsed_s": [0., 5., 10.], "T.temp_external": [25., 23., 20.],
                          "S.speed_rpm": [100., 200., 300.], "AS.flow": [6., 6., 6.]}).to_csv(run / "execution.csv", index=False)
            settings = {"enabled": True, "power_number": 2., "diameter_m": .02, "volume_mode": "charged", "volume_ml": 100.}
            exps, inputs, _ = read_completed_recordings(folder, (definition.experiment_id,),
                {**default_setup()["recording"], "initial_concentration": .1}, {"Temperature": "T.temp_external"},
                {definition.experiment_id: process()}, {definition.experiment_id: (0., 10.)},
                roles={"T": "T", "stirring_rate": "S", "AS": "AS"}, mixing=settings)
            self.assertEqual(inputs[0]["StirringRPM"], 200.)
            self.assertEqual(inputs[0]["MixingSummary"]["average_volume_ml"], 100.5)
            self.assertAlmostEqual(exps[0]["AddedAntisolventData"][1, -1], 2.)
            self.assertIn("Calibrated pump", exps[0]["AntisolventSource"])
            self.assertNotIn("StirringRateData", exps[0])

    def test_worker_stops_before_understocked_second_run_and_preserves_first_status(self):
        from experiment_dashboard.experiment_dashboard import ExperimentWorker
        from hardware_devices.simulated import SimulatedDevice
        topology, configs = fixture()
        first = ExperimentDefinition("First", (profile("P", [(0, 60), (10, 0)], True),), initial_mass_g=20.)
        second = ExperimentDefinition("Second", first.profiles, initial_mass_g=20.)
        book = ReservoirBook()
        book.configure(topology, configs)
        pump = SimulatedDevice("P", "ismatec", noise=0.)
        pump.connect()
        campaign = Campaign("Stock test", (first, second))
        worker = ExperimentWorker({"P": pump}, {"P": configs["P"]}, list(campaign.experiments),
                                  speed=100., end_hold_s=0., campaign=campaign, reservoirs=book)
        packets = []
        worker.status_updated.connect(packets.append)
        worker.start_experiment()
        self.assertEqual(worker.outcome.value, "failed")
        self.assertIn("Insufficient reservoir stock", worker.error_message)
        self.assertEqual([run.status.value for run in worker.runs], ["completed", "failed"])
        self.assertAlmostEqual(book.remaining["feed"], 5.)
        self.assertTrue(any(packet.get("system_mass_g") is not None for packet in packets))
        pump.disconnect()

    def test_old_bo_save_generates_stirring_stages_on_reload(self):
        from bo.window import BOWindow
        from bo.settings import empty_settings
        state = empty_settings()
        state["tolerances"].pop("volume")  # Older save files predate this optional input.
        for index, rpm in enumerate((200., 300., 400.)):
            state["runs"].append({"name": f"Run {index}", "profiles": {
                "temperature": Profile(np.array([0., 10.]), np.array([25., 20.]), "T").payload(),
                "stirring": Profile(np.array([0., 10.]), np.array([rpm, rpm]), "Stirrer").payload()},
                "metrics": {"duration": 10., "d50": 20. + index,
                            **dict.fromkeys(("yield_actual", "yield_theoretical", "d10", "d90"))}})
        state["selected"] = [0, 1, 2]
        window = BOWindow()
        try:
            window.restore(state)
            self.assertIn("stirring", window.state["physical_profiles"])
            editor = window.stage_editor.editors["stirring"]
            self.assertEqual(editor.table.rowCount(), 1)
            self.assertGreater(editor.controls.rowCount(), 0)
            editor.table.item(0, 2).setText("250")
            editor.table.item(0, 3).setText("250")
            saved = window.settings()
            self.assertEqual(saved["physical_profiles"]["stirring"]["stages"][0]["start"], 250.)
        finally:
            window.close()

    def test_stirring_import_keeps_final_hold_and_portable_commands(self):
        from bo.profiles import campaign_profiles, case_runs
        from experiment.recording import atomic_json
        _, configs = fixture()
        configs["S2"] = {**deepcopy(configs["S"]), "connectivity": {"unit": "feed"}}
        spec = ExperimentDefinition("Run", (profile("T", [(0, 25), (10, 20)]),
             DeviceProfile("S", (ProfilePoint(0., 100.), ProfilePoint(10., 300.)), unit="RPM"),
             DeviceProfile("S2", (ProfilePoint(0., 500.), ProfilePoint(10., 500.)), unit="RPM")))
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            recording = Path(folder) / "execution.csv"
            atomic_json(recording.with_name("run.json"), {"experiment_id": spec.experiment_id})
            experiment = {"RecordingFile": str(recording), "TemperatureData": [[0., 15.], [25., 20.]]}
            definition = {"campaign": asdict(Campaign("Custom", (spec,), line_roles={
                "custom_mixer": {"role": "stirring_rate", "device_id": "S"}})), "devices": configs}
            found = campaign_profiles(experiment, process(), definition)
            self.assertEqual(found["stirring"].time[-1], 15.)
            self.assertEqual(found["stirring"].values[-1], 300.)
            experiment["CommandedProfiles"] = {key: value.payload() for key, value in found.items()}
            experiment["RecordingFile"] = str(Path(folder) / "moved-away" / "execution.csv")
            recovered = case_runs({"ExpData": [experiment], "ProcInf": [process()]})
            self.assertEqual(recovered[0]["profiles"]["stirring"]["points"][1][-1], 300.)

    def test_dashboard_uses_msi_and_tracks_stock_across_bo_followup(self):
        from test_adaptive_loop import AdaptiveLoopTests, PRESET
        from imed_python.demo_campaign import model_inputs
        checks = AdaptiveLoopTests()
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            window, output, _, logs = checks.setup_dashboard(Path(directory))
            try:
                topology, configs = fixture(stock=100.)
                from uuid import uuid4
                topology["unit_operations"][0]["inventory_id"] = uuid4().hex
                panel = window.control_panel
                panel.apply_loaded_configuration(configs, {}, [], system_topology=topology)
                for key, device in panel.hardware.items():
                    device.connect()
                    panel.device_widgets[key]._update_connection_state()
                panel.simulation_speed.setValue(100.)
                frames = []
                for temperature, rpm in ((14., 100.), (19., 200.), (24., 300.)):
                    frame = pd.DataFrame({"time_s": [0., np.nextafter(5., 0.), 5., 10., 15.],
                        "T": [temperature] * 5, "S": [rpm] * 5, "P": [60., 60., 0., 0., 0.], "V": [0., 0., 0., 1., 0.]})
                    frame.attrs["model_window"] = [5., 10.]
                    frames.append(frame)
                panel.experiments = frames
                _, _, base, solubility, _ = model_inputs(False)
                window.imed_page.panel.restore_setup_settings({"Solubility": solubility})
                setup = window.settings_page.loop_setup
                setup["manual_assignments"] = {}
                setup["process"] = base
                setup["recording"].update(no_feed=True, initial_concentration=.1)
                setup["channels"] = {"Concentration": ""}
                bo = json.loads(PRESET.with_name("bo_validation_loop.json").read_text())
                bo["targets"]["d50"]["enabled"] = False
                bo["targets"]["yield"]["enabled"] = True
                bo["physical_profiles"] = {}
                window.bo_page.restore(bo)
                from unittest.mock import patch
                with patch.object(window, "_choose_loop_source", return_value="manual"):
                    window.start_command()
                self.assertIsNotNone(window._adaptive_loop, logs)
                self.assertTrue(checks.wait_for(lambda: window._adaptive_loop is None, timeout=60), logs)
                self.assertFalse(any(level == "ERROR" for level, _ in logs), logs)
                self.assertEqual(len(window.bo_page.state["runs"]), 4)
                self.assertAlmostEqual(panel.reservoirs.remaining["feed"], 80., places=5)
                self.assertIn("80.000 g", window.dashboard_page.reservoir_values.text())
                files = list(output.glob("campaign_*/*/execution.csv"))
                self.assertEqual(len(files), 4)
                self.assertTrue(all("system_mass_g" in pd.read_csv(file, nrows=1).columns for file in files))
                self.assertIn("stirring", window.bo_page.state["initial_input_groups"])
                self.assertIn("stirring", window.bo_page.state["physical_profiles"])
            finally:
                checks.doCleanups()
