"""Analytical averages, kinetic equivalence and independent PBM convergence."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiment.adaptive_loop import default_setup, read_completed_recordings
from experiment.models import Campaign, DeviceProfile, ExperimentDefinition, ProfilePoint
from experiment.recording import atomic_json
from imed_python.distribution import cell_volumes
from imed_python.mixing import DEFAULT_MIXING, average_mixing, trace_mean
from imed_python.solver import simulate
from imed_python.validation.cases import Case
from imed_python.validation.qmom import solve_ccas_qmom


MIXING = {**DEFAULT_MIXING, "enabled": True, "power_number": 2., "diameter_m": .04, "volume_ml": 100.}


def kinetics(cells, *, secondary=True, aggregation=True):
    pm, exp, proc, sol, settings = Case("cooling", "growth").inputs(cells)
    if secondary:
        pm[5:12] = [5., 1., 0., 2., -np.inf, 0., 1.2]
    if aggregation:
        pm[24:28] = [np.log10(5e-8), 0., 1., .5]
    proc = average_mixing(exp, proc, MIXING, mean_rpm=300.)
    return pm, exp, proc, sol, settings


class AverageMixingTests(unittest.TestCase):
    def test_units_and_independent_rpm_volume_scaling(self):
        _, exp, proc, _, _ = kinetics(20)
        baseline = average_mixing(exp, proc, MIXING, mean_rpm=300.)
        self.assertAlmostEqual(baseline["PowerRatio"], .256, places=14)
        doubled_rpm = average_mixing(exp, proc, MIXING, mean_rpm=600.)
        doubled_volume = average_mixing(exp, proc, {**MIXING, "volume_ml": 200.}, mean_rpm=300.)
        self.assertAlmostEqual(doubled_rpm["PowerRatio"] / baseline["PowerRatio"], 8.)
        self.assertAlmostEqual(doubled_volume["PowerRatio"] / baseline["PowerRatio"], .5)
        self.assertEqual(average_mixing(exp, proc, {**MIXING, "enabled": False})["PowerRatio"], 1.)

    def test_linear_and_step_means_are_time_weighted(self):
        points = [[0., 5., 10.], [100., 300., 300.]]
        self.assertEqual(trace_mean(points, 10., "RPM", step=True), 200.)
        self.assertEqual(trace_mean(points, 10., "RPM"), 250.)
        self.assertEqual(trace_mean([[0, 1, 10], [100, 300, 300]], 10., "RPM", step=True), 280.)

    def test_charged_average_volume_matches_analytic_ramped_addition(self):
        exp = {"TemperatureData": [[0., 10.], [25., 20.]], "FlowRateData": [[0., 10.], [0., .2]]}
        proc = {"SolvMass": 100., "AntiSolvMass": 0., "SolventDensity": 1., "AntiSolventDensity": 1.}
        settings = {**MIXING, "volume_mode": "charged"}
        result = average_mixing(exp, proc, settings, mean_rpm=300.)
        # q(t)=0.02t g/s; mean cumulative addition = 0.02 T²/6.
        expected_volume = 100. + 1 / 3
        self.assertAlmostEqual(result["MixingSummary"]["average_volume_ml"], expected_volume, places=12)
        self.assertAlmostEqual(result["PowerRatio"], 2 * 5**3 * .04**5 / (expected_volume * 1e-6), places=12)
        time = np.linspace(0., 10., 101)
        exp["AddedAntisolventData"] = [time, .01 * time**2]
        recorded = average_mixing(exp, proc, settings, mean_rpm=300.)
        self.assertLess(abs(recorded["PowerRatio"] / result["PowerRatio"] - 1), 2e-7)

    def test_commanded_step_rpm_import_uses_holds_instead_of_ramps(self):
        import pandas as pd
        profiles = (DeviceProfile("T", (ProfilePoint(0., 25.), ProfilePoint(10., 20.))),
                    DeviceProfile("S", (ProfilePoint(0., 100.), ProfilePoint(5., 300.), ProfilePoint(10., 300.)), "step", "RPM"))
        definition = ExperimentDefinition("Step stirring", profiles)
        configs = {"T": {"type": "huber"}, "S": {"type": "jkem"}}
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            folder = Path(directory)
            atomic_json(folder / "definition.json", {"campaign": asdict(Campaign("Step", (definition,))), "devices": configs})
            run = folder / "run1"
            run.mkdir()
            atomic_json(run / "run.json", {"experiment_id": definition.experiment_id, "experiment_number": 1,
                                           "status": "completed", "recording_complete": True})
            pd.DataFrame({"elapsed_s": [0., 5., 10.], "T.temp_external": [25., 22., 20.]}).to_csv(run / "execution.csv", index=False)
            proc = {"SolvMass": 100., "AntiSolvMass": 0., "SolventDensity": 1., "AntiSolventDensity": 1., "SeedMass": 0.}
            recording = {**default_setup()["recording"], "no_feed": True, "initial_concentration": .1}
            _, processes, _ = read_completed_recordings(folder, [definition.experiment_id], recording,
                {"Temperature": "T.temp_external"}, {definition.experiment_id: proc}, {}, roles={"T": "T", "stirring_rate": "S"}, mixing=MIXING)
            self.assertEqual(processes[0]["StirringRPM"], 200.)

    def test_sensitivity_recomputes_power_from_scaled_rpm_and_initial_volume(self):
        from imed_python.probabilistic import _trial
        pm, exp, proc, sol, settings = kinetics(20)
        proc.update(StirringRPM=300., InitialVolume_ml=100., MixingSettings={**MIXING, "volume_mode": "charged"})
        baseline = average_mixing(exp, proc, proc["MixingSettings"], mean_rpm=300.)["PowerRatio"]
        with patch("imed_python.probabilistic._predict", side_effect=lambda p, e, proc, *args, **kw: proc["PowerRatio"]):
            value, error = _trial(np.array([2., 2.]), baseline=pm, indices=np.array([], dtype=int),
                condition_keys=["stirring_factor", "volume_factor"], experiment=exp, process=proc, solubility=sol, settings=settings)
        self.assertEqual(error, "")
        self.assertAlmostEqual(value / baseline, 4.)

    def test_secondary_and_agglomeration_match_exact_prefactor_rescaling(self):
        for secondary, aggregation in ((True, False), (False, True), (True, True)):
            with self.subTest(secondary=secondary, aggregation=aggregation):
                pm, exp, proc, sol, settings = kinetics(80, secondary=secondary, aggregation=aggregation)
                settings.update(RelTol=1e-9, AbsTol=1e-11, MaxStep=.1)
                adjusted = pm.copy()
                if secondary:
                    adjusted[5] += pm[11] * np.log10(proc["PowerRatio"])
                if aggregation:
                    adjusted[24] += pm[27] * np.log10(proc["PowerRatio"])
                actual = simulate(pm, exp, proc, sol, settings)
                reference = simulate(adjusted, exp, {**proc, "PowerRatio": 1.}, sol, settings)
                for a, b in zip(actual, reference):
                    np.testing.assert_array_equal(a[0], b[0])
                    scale = max(float(np.max(np.abs(b[1]))), 1e-12)
                    self.assertLess(float(np.max(np.abs(a[1] - b[1]))) / scale, 1e-6)

    def test_primary_nucleation_remains_independent_of_the_secondary_power_term(self):
        pm, exp, proc, sol, settings = Case("cooling", "nucleation").inputs(40)
        a = simulate(pm, exp, {**proc, "PowerRatio": .01}, sol, settings)
        b = simulate(pm, exp, {**proc, "PowerRatio": 10.}, sol, settings)
        for actual, expected in zip(a, b):
            np.testing.assert_allclose(actual, expected, rtol=0., atol=0.)

    def test_power_coupled_mesh_refinement_and_api_mass_conservation(self):
        errors, rows = [], []
        for cells in (80, 160, 320):
            pm, exp, proc, sol, settings = kinetics(cells)
            count, concentration, _, state = simulate(pm, exp, proc, sol, settings, return_history=True)
            reference = solve_ccas_qmom(pm, exp, proc, sol, settings, time=count[0])
            reference_volume = proc["SolvMass"] / proc["SolventDensity"] + (reference.liquid_mass - proc["SolvMass"]) / proc["AntiSolventDensity"]
            reference_count = reference.number / reference_volume
            error = float(np.max(np.abs(count[1] - reference_count)) / np.max(reference_count))
            errors.append(error)
            edges = np.linspace(settings["Lmin"], settings["Lmax"], cells + 1)
            solid = (cell_volumes(edges, proc["ShapeFactor"]) * np.diff(edges)) @ state[:cells] * proc["SolidDensity"] / 1e12
            api = solid + state[cells] * state[cells + 1] + state[cells + 5]
            mass_error = float(np.max(np.abs(api - api[0])) / api[0])
            self.assertLess(mass_error, 1e-8)
            self.assertTrue(np.isfinite(state).all())
            self.assertEqual(state.shape[0], cells + 6)
            self.assertTrue(np.all(count[1] >= 0))
            self.assertTrue(np.all(concentration[1] >= 0))
            rows.append({"cells": cells, "epsilon": proc["PowerRatio"], "count_error_vs_qmom": error, "mass_relative_error": mass_error})
        self.assertLess(errors[-1], .01)
        self.assertLessEqual(errors[-1], max(errors[0] * 1.05, .002))
        print("MIXING_CONVERGENCE " + json.dumps(rows), flush=True)

    def test_zero_power_has_an_actionable_error_for_negative_active_exponents(self):
        pm, exp, proc, sol, settings = kinetics(20)
        pm[11] = -1.
        with self.assertRaisesRegex(ValueError, "Zero mixing power"):
            simulate(pm, exp, {**proc, "PowerRatio": 0.}, sol, settings)
        pm[11] = 1.
        count, _, _ = simulate(pm, exp, {**proc, "PowerRatio": 0.}, sol, settings)
        self.assertTrue(np.isfinite(count).all())
