"""Check shared DoE decisions through BO, IMED design and sensitivity entry points."""
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from bo.profiles import Profile, case_runs
from experiment.adaptive_loop import default_setup, validate_setup
from experiment.doe_decisions import case_decisions, decision_groups, planned_decisions
from experiment.models import Campaign, DeviceProfile, ExperimentDefinition, ProfilePoint
from imed_python.design import optimize_recipe
from imed_python.design_dialog import DesignDialog
from imed_python.doe_design import imed_recipe_profiles, seed_template
from imed_python.model_file import save_checkpoint
from imed_python.native_case import pack_case, unpack_case
from imed_python.probabilistic import propagate
from imed_python.probabilistic_dialog import ProbabilisticDialog
from imed_python.solver import read_model
from imed_python import test_probabilistic


APP = QApplication.instance() or QApplication([])


def case_with_doe(*, vary_temperature=False, vary_feed=False, vary_seed=True, vary_volume=True):
    case = test_probabilistic.ProbabilisticTests.case()
    model = case["ModelParameter"][0]
    model["paraset"][5, 1] = 5.
    model["paraset"][11, 1] = .5
    model["para"] = model["paraset"][model["paraset"][:, 0] == 1, 1].copy()
    case["ModelParameter"].append(deepcopy(model))
    case["ModelParameter"][1]["paraset"][11, 1] = 0.
    base, process = case["ExpData"][0], case["ProcInf"][0]
    case["ExpData"], case["ProcInf"] = [], []
    for index, rpm in enumerate((120., 180., 240.)):
        temperature = Profile(np.array([0., 2., 5.]), np.array([25., 24. - index * vary_temperature, 24. - index * vary_temperature]), "Command · T")
        feed = Profile(np.array([0., 1., 2., 5.]), np.array([0., .02 + index * .01 * vary_feed, 0., 0.]), "Command · AS")
        stirrer = Profile(np.array([0., 5.]), np.full(2, rpm), "Command · S")
        case["ExpData"].append({**base, "TemperatureData": np.vstack((temperature.time, temperature.values + .01 * index)),
                               "FlowRateData": np.vstack((feed.time, feed.values)), "FinalD10D50D90": [11., 20. + index, 30.],
                               "CommandedProfiles": {key: value.payload() for key, value in
                                                     (("temperature", temperature), ("feed", feed), ("stirring", stirrer))}})
        case["ProcInf"].append({**process, "StirringRPM": rpm, "InitialVolume_ml": 100. + index * 20. * vary_volume,
                               "SeedMass": .02 + .01 * index * vary_seed})
    return case


class DoeDecisionTests(unittest.TestCase):
    def test_planned_starting_batch_saves_and_ignores_fixed_nonzero_as(self):
        def profile(device, targets, unit):
            return DeviceProfile(device, tuple(ProfilePoint(time, target) for time, target in zip((0., 2., 5.), targets)), "linear", unit)

        experiments = tuple(ExperimentDefinition(str(index), (profile("T", (25., 24., 24.), "°C"),
                            profile("A", (0., 1.2, 0.), "mL/min"), profile("S", (rpm, rpm, rpm), "RPM")),
                            seed_mass_g=.02 + .01 * index) for index, rpm in enumerate((120., 180., 240.)))
        setup = default_setup()
        setup["process"].update(SolvMass=100., SolventDensity=1., AntiSolventDensity=1.)
        setup["initial_processes"] = {"1": {"SolvMass": 120.}, "2": {"SolvMass": 140.}}
        configs = {"T": {"type": "huber"}, "A": {"type": "ismatec", "simulated": True}, "S": {"type": "jkem"}}
        campaign = Campaign("DoE", experiments, line_roles={"feed_lane": {"role": "AS", "device_id": "A"}})
        space = planned_decisions(campaign, setup, {"T": "T", "feed_lane": "A", "stirring_rate": "S"}, configs, {})
        self.assertEqual(decision_groups(space), {"stirring", "volume", "seed"})
        self.assertEqual(len(space["traces"]["feed"]), 1)
        self.assertAlmostEqual(space["traces"]["feed"][0]["points"][1][1], .02)
        self.assertEqual(space["levels"]["volume"], [100., 120., 140.])
        setup["doe_decisions"] = space
        self.assertEqual(validate_setup(setup)["doe_decisions"], space)

    def test_offline_bo_preserves_the_loaded_plan_with_later_runs(self):
        from bo.optimizer import propose
        from bo.window import BOWindow

        case = case_with_doe(vary_seed=False, vary_volume=False)
        page = BOWindow()
        try:
            page.set_case(case, "Offline case")
            state = page.settings()
            state["targets"]["d50"]["enabled"] = True
            self.assertEqual(decision_groups(state["doe_decisions"]), {"stirring"})
            self.assertEqual(page.stage_editor.editors["feed"].control_note.text(), "Fixed in the starting DoE.")
            followup = case_runs(case_with_doe(vary_temperature=True, vary_feed=True))[2]
            state["selected"].append(len(state["runs"]))
            state["runs"].append(followup)
            result = propose(state)
            vary = {variable.key.split(":")[0] for variable, enabled in zip(result.variables, result.vary) if enabled}
            self.assertEqual(vary, {"stirring"})
            np.testing.assert_allclose(result.profiles["temperature"].sample([0, 2, 5]), [25., 24., 24.])
            np.testing.assert_allclose(result.profiles["feed"].sample([0, 1, 2, 5]), [0., .02, 0., 0.])
        finally:
            page.close()

    def test_bo_can_fit_preparation_effects_with_followup_doses_held_fixed(self):
        from bo.optimizer import propose
        from bo.settings import empty_settings

        runs = case_runs(case_with_doe())
        state = empty_settings()
        state.update(runs=runs, selected=list(range(len(runs))))
        state["targets"]["d50"]["enabled"] = True
        result = propose(state, fixed_inputs={"seed": .03, "volume": 120.})
        for key, expected in (("seed", .03), ("volume", 120.)):
            column = next(i for i, var in enumerate(result.variables) if var.key == f"{key}:scalar")
            self.assertGreater(np.ptp(result.observed[:, column]), 0)
            self.assertFalse(result.vary[column])
            self.assertAlmostEqual(result.profiles[key].values[0], expected)

    def test_loaded_imed_case_drives_md_do_and_power_qualified_rpm(self):
        case = case_with_doe(vary_temperature=True)
        before = case_decisions(case)
        case.setdefault("PythonWorkflow", {})["doe_decisions"] = before
        loaded = unpack_case(pack_case(case))
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            matlab_case = Path(directory) / "doe.mat"
            save_checkpoint(None, matlab_case, case)
            self.assertEqual(case_decisions(read_model(matlab_case)), before)
        experiment, process = loaded["ExpData"][0], loaded["ProcInf"][0]
        for optimal in (False, True):
            with self.subTest(optimal=optimal):
                dialog = DesignDialog(loaded, optimal)
                try:
                    controls = dialog.decision_controls(experiment, process)
                    self.assertEqual(set(controls), {"temperature", "stirring", "volume", "seed"})
                    if optimal:
                        dialog.model.setCurrentIndex(1)
                        self.assertNotIn("stirring", dialog.decision_controls(experiment, process))
                    request = {"doe_search": {"space": before, "controls": controls, "seed_template": seed_template(loaded)}}
                    _, optimized, adjusted = optimize_recipe(experiment, process, request,
                        lambda exp, proc: exp["ProposedStirringRPM"] + proc["InitialVolume_ml"], iterations=2)
                    profiles = imed_recipe_profiles(optimized, "Test proposal")
                    self.assertGreater(profiles["stirring"].values[0], 180.)
                    self.assertGreater(adjusted["InitialVolume_ml"], 120.)
                    np.testing.assert_allclose(profiles["feed"].sample([0, 1, 2, 5]), [0., .02, 0., 0.])
                finally:
                    dialog.close()

    def test_offline_sa_enables_varying_groups_and_compares_profile_choices(self):
        case = case_with_doe(vary_temperature=True, vary_feed=False)
        dialog = ProbabilisticDialog(case, sensitivity=True)
        try:
            enabled = {key for key, (control, _) in dialog.doe_checks.items() if control.isEnabled() and control.isChecked()}
            self.assertEqual(enabled, {"temperature", "stirring", "volume", "seed"})
            dialog.model.setCurrentIndex(1)
            self.assertFalse(dialog.doe_checks["stirring"][0].isEnabled())
        finally:
            dialog.close()

        def prediction(_parameters, experiment, process, *_args, **_kwargs):
            np.testing.assert_allclose(experiment["FlowRateData"], [[0., 1., 2., 5.], [0., .02, 0., 0.]])
            return {"percentiles": np.full((3, 2), experiment["TemperatureData"][1, -1]),
                    "time": np.array([0., 5.]), "counts": np.ones(2), "concentration": np.full(2, process["PowerRatio"])}

        with patch("imed_python.probabilistic._predict", side_effect=prediction):
            result = propagate(case, 0, 0, samples=24, use_parameters=False, doe_conditions=["temperature", "stirring", "volume"])
            self.assertEqual(result["input_names"], ["Temperature profile choice", "Stirring rate", "Starting liquid volume"])
            self.assertEqual(len(result["choice_effects"]), 3)
            self.assertTrue(np.isnan(result["prcc"][0]).all())
            self.assertGreater(np.ptp(result["concentrations"][:, -1]), 0.)
            with self.assertRaisesRegex(ValueError, "fixed"):
                propagate(case, 0, 0, samples=4, use_parameters=False, doe_conditions=["feed"])

    def test_changed_as_with_a_fixed_initial_charge_does_not_enable_volume(self):
        case = case_with_doe(vary_feed=True, vary_volume=False, vary_seed=False)
        for index, process in enumerate(case["ProcInf"]):
            process["AverageVolume_ml"] = 110. + index
        groups = decision_groups(case_decisions(case))
        self.assertEqual(groups, {"feed", "stirring"})

    def test_offline_campaign_import_recovers_differing_preparation_charges(self):
        import pandas as pd
        from experiment.recording import atomic_json
        from imed_python.campaign_dialog import CampaignImportDialog
        from test_linked_inputs import fixture, process, profile

        _, configs = fixture()
        experiments = [ExperimentDefinition(f"Charge {i}", (
            profile("T", [(0, 25), (10, 25)]), profile("P", [(0, rate), (5, 0), (10, 0)], True)),
            model_window=(5., 10.)) for i, rate in enumerate((60, 120))]
        campaign = Campaign("Offline DoE", tuple(experiments), line_roles={"charge": {"role": "solvent", "device_id": "P"}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            folder = Path(directory)
            atomic_json(folder / "definition.json", {"campaign": asdict(campaign), "devices": configs})
            runs = []
            for index, experiment in enumerate(experiments, 1):
                run = folder / str(index)
                run.mkdir()
                atomic_json(run / "run.json", {"experiment_id": experiment.experiment_id})
                file = run / "execution.csv"
                pd.DataFrame({"elapsed_s": [0, 5, 7, 10], "T.temp_external": [25, 25, 25, 25]}).to_csv(file, index=False)
                runs.append((index, experiment.name, file))
            dialog = CampaignImportDialog(runs, {"ProcInf": [process()]}, None)
            try:
                dialog.no_feed.setChecked(True)
                dialog.initial_concentration.setValue(.1)
                dialog.channels["Temperature"].setCurrentIndex(dialog.channels["Temperature"].findData("T.temp_external"))
                with patch("imed_python.campaign_dialog.QMessageBox.warning") as warning:
                    dialog.import_selection()
                    warning.assert_not_called()
                case = {"ExpData": dialog.experiments, "ProcInf": dialog.processes}
                self.assertEqual([item["InitialVolume_ml"] for item in dialog.processes], [5., 10.])
                self.assertEqual(decision_groups(case_decisions(case, folder)), {"volume"})
            finally:
                dialog.close()

    def test_imed_recipe_passes_new_rpm_charge_and_observed_seed_to_devices(self):
        from experiment.adaptive_loop import RecipeCompiler
        from experiment.doe_campaign import ProfileAssignment
        from test_linked_inputs import fixture, process, profile

        _, configs = fixture()
        configs["SEED"] = {"name": "SEED", "type": "masterflex", "simulated": True}
        runs = []
        for seed in (.02, .04):
            profiles = (profile("T", [(0, 25), (5, 25), (10, 24)]),
                        profile("P", [(0, 60), (5, 0), (10, 0)], True),
                        profile("S", [(0, 120), (5, 120), (10, 120)]),
                        profile("SEED", [(0, 0), (3, 1000 * seed), (4, 0), (10, 0)], True))
            runs.append(ExperimentDefinition("Starting recipe", profiles, model_window=(5., 10.), seed_mass_g=seed))
        setup = default_setup()
        assignments = tuple(ProfileAssignment(role, device) for role, device in
                            (("T", "T"), ("solvent", "P"), ("stirring_rate", "S"), ("seed", "SEED")))
        compiler = RecipeCompiler(setup, Campaign("DoE", tuple(runs)), configs, None, assignments, (), "imed")
        case = case_with_doe()
        experiment = {**case["ExpData"][0], "ProposedStirringRPM": 210.,
                      "FlowRateData": [[0., 5.], [0., 0.]]}
        inputs = {**process(), "SeedMass": .04, "SolvMass": 8., "InitialVolume_ml": 8.,
                  "RequestedInitialVolume_ml": 8., "VolumeFromDesignConditions": True}
        result, window = compiler.build(imed_recipe_profiles(experiment, "IMED proposal"), inputs,
                                       name="Follow-up", source="imed", default_seed=.02)
        device_profiles = {target.device_id: target for target in result.experiments[0].profiles}
        self.assertEqual(window, (5., 10.))
        self.assertEqual(device_profiles["P"].points[0].target, 96.)
        self.assertEqual(device_profiles["SEED"].points[1].target, 40.)
        self.assertEqual(device_profiles["S"].points[-1].target, 210.)
