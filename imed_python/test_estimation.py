import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import differential_evolution

from imed_python.distribution import d_values, volume_distribution
from imed_python.estimation import FitObjective, fit_model
from imed_python.model_file import save_model
from imed_python.solver import read_model, simulate
from imed_python.size_data import FINAL_SIZE_TERM, TREND_SIZE_TERM


class EstimationTests(unittest.TestCase):
    def test_trend_objective_compares_all_recorded_size_samples_at_their_times(self):
        experiment = {"TemperatureData": [[0., 10.], [20., 20.]],
                      "FinalD10D50D90": [20., 40., 60.],
                      "D10Data": [[1., 9.], [10.5, 19.]],
                      "D50Data": [[0., 10.], [20., 38.]],
                      "D90Data": [[5.], [44.]]}
        outputs = (np.array([[0., 5., 10.], [1., 2., 3.]]), None,
                   np.array([[.5, 1.5, 2.5], [1., 1., 1.]]))
        predicted = np.array([[10., 15., 20.], [20., 30., 40.], [30., 45., 60.]])
        objective = FitObjective([experiment], [{"ShapeFactor": 1}], {}, {"N": 3}, [0, 0, 1], TREND_SIZE_TERM)
        self.assertEqual(objective.residual_count, 5)
        self.assertEqual(objective.sample_count, 5)
        with patch("imed_python.estimation.simulate", return_value=(*outputs, np.ones((9, 3)))) as solve:
            with patch("imed_python.comparison.percentile_history", return_value=predicted):
                np.testing.assert_allclose(objective.residuals(np.zeros(35)), np.array([.5, 0, 0, 2, 1])/np.sqrt(5))
        self.assertTrue(solve.call_args.kwargs["return_history"])
        final = FitObjective([experiment], [{"ShapeFactor": 1}], {}, {"N": 3}, [0, 0, 1], FINAL_SIZE_TERM)
        self.assertEqual(final.residual_count, 3)
        with patch("imed_python.estimation.simulate", return_value=outputs):
            with patch("imed_python.estimation.d_values", return_value=[20., 40., 60.]):
                np.testing.assert_array_equal(final.residuals(np.zeros(35)), np.zeros(3))

    def test_trend_objective_retains_older_final_only_experiments(self):
        experiment = {"TemperatureData": [[0., 10.], [20., 20.]], "FinalD10D50D90": [20., 40., 60.]}
        objective = FitObjective([experiment], [{"ShapeFactor": 1}], {}, {"N": 3}, [0, 0, 1], TREND_SIZE_TERM)
        self.assertEqual(objective.residual_count, 3)
        for column, points in objective.experiments[0][2][2].items():
            np.testing.assert_array_equal(points, [[10.], [experiment["FinalD10D50D90"][column]]])

    def test_trend_fit_recovers_a_rate_from_synthetic_size_histories(self):
        from imed_python.comparison import percentile_history
        from imed_python.validation.cases import Case

        parameters, experiment, process, solubility, settings = Case("cooling", "growth").inputs(20)
        count, _, csd, state = simulate(parameters, experiment, process, solubility, settings, return_history=True)
        values = percentile_history(csd, state, process, settings)
        for column, field in enumerate(("D10Data", "D50Data", "D90Data")):
            experiment[field] = np.vstack((count[0, ::20], values[column, ::20]))
        definitions = np.column_stack((np.zeros(35), parameters, parameters, parameters))
        definitions[12] = [1, -.3, -.6, 0.]
        objective = FitObjective([experiment], [process], solubility, settings, [0, 0, 1], TREND_SIZE_TERM)
        result = fit_model({"paraset": definitions}, objective, global_iterations=0, local_evaluations=12)
        np.testing.assert_allclose(result.parameters, parameters[[12]], atol=1e-4)
        self.assertLess(result.ssr, 1e-8)

    def test_count_normalization_follows_each_matlab_stage(self):
        observed = [[0, 1, 2], [3, 12, 6]]
        predicted = np.array([[0, 1, 2], [2, 8, 2]])
        objective = FitObjective([{"CountsData": observed}], [{"ShapeFactor": 1}],
                                 {}, {"N": 4}, [1, 0, 0], "CSD")
        with patch("imed_python.estimation.simulate", return_value=(predicted, None, None)):
            np.testing.assert_allclose(objective.residuals(np.zeros(35)),
                                       np.array([0, 0, -0.25]) / np.sqrt(3))
            np.testing.assert_allclose(objective.residuals(np.zeros(35), global_stage=True),
                                       np.array([0.5, 2, 0]) / np.sqrt(3))

    def test_d_values_on_uniform_distribution(self):
        np.testing.assert_allclose(d_values([0, 100], [1, 1]), [10, 50, 90])

    def test_volume_distribution_weights_larger_particles_and_normalizes_area(self):
        length = np.array([0.5, 1.5, 2.5])
        process = {"ShapeFactor": 1}
        settings = {"Lmin": 0, "Lmax": 3}
        distribution, percentiles = volume_distribution(
            np.vstack((length, [1, 1, 1])), process, settings)
        self.assertAlmostEqual(np.trapezoid(distribution[1], length), 100)
        self.assertGreater(percentiles[1], 1.5)
        empty, no_percentiles = volume_distribution(
            np.vstack((length, [0, 0, 0])), process, settings)
        self.assertIsNone(no_percentiles)
        np.testing.assert_array_equal(empty[1], 0)

    def test_fit_recovers_nucleation_rate_from_synthetic_concentration(self):
        parameters = np.zeros(35)
        parameters[[5, 18, 24, 28]] = -np.inf
        parameters[[0, 12]] = [6, -0.5]
        experiment = {"TemperatureData": [[0, 30], [20, 20]], "ConcentrationData": [[0], [0.1]]}
        process = {"ShapeFactor": 1, "SolidDensity": 1.4, "SolventDensity": 0.86,
                   "AntiSolventDensity": 1, "PowerRatio": 1, "SeedMass": 0,
                   "SolvMass": 10, "AntiSolvMass": 0}
        solubility = dict.fromkeys("abcdefgh", 0)
        solubility["a"] = np.log(0.05)
        settings = {"N": 20, "Lmax": 100}
        _, concentration, _ = simulate(parameters, experiment, process, solubility, settings)
        experiment["ConcentrationData"] = concentration[:, ::5]
        objective = FitObjective([experiment], [process], solubility, settings, [0, 1, 0], "CSD")
        definitions = np.column_stack((np.zeros(35), parameters, parameters, parameters))
        definitions[0] = [1, 5, 4, 8]
        initial = parameters.copy()
        initial[0] = 5
        initial_errors = objective.residuals(initial, global_stage=True)
        initial_score = float(initial_errors @ initial_errors)
        progress = []
        with patch("imed_python.estimation.differential_evolution", wraps=differential_evolution) as fit:
            result = fit_model({"paraset": definitions}, objective,
                               global_iterations=3, population_size=6, global_tolerance=1e-6,
                               mutation_min=0.4, mutation_max=0.9, recombination=0.6,
                               progress=progress.append)
        self.assertEqual(fit.call_args.kwargs["init"].shape, (6, 1))
        self.assertEqual(fit.call_args.kwargs["tol"], 1e-6)
        self.assertEqual(fit.call_args.kwargs["mutation"], (0.4, 0.9))
        self.assertEqual(fit.call_args.kwargs["recombination"], 0.6)
        self.assertTrue(any(line.startswith("Differential evolution · 1 trial evaluation ·")
                            and f"current objective {initial_score:.5g}" in line for line in progress))
        self.assertTrue(any("generation 1/3" in line and "best WMSE" in line for line in progress))
        self.assertTrue(any(line.startswith("Local least-squares refinement")
                            and "current objective" in line for line in progress))
        self.assertTrue(result.converged)
        self.assertGreater(result.simulation_count, 0)
        self.assertGreater(result.simulation_seconds, 0)
        self.assertTrue(any("Simulation timing" in line and "s/run" in line for line in progress))
        self.assertIsNotNone(result.intervals, result.uncertainty_error)
        self.assertEqual(result.intervals.convention, "matlab")
        np.testing.assert_allclose(result.intervals.covariance, result.covariance)
        np.testing.assert_allclose(result.parameters, [6], atol=0.01)
        self.assertLess(result.ssr, 1e-9)

        parallel_progress = []
        parallel = fit_model({"paraset": definitions}, objective, workers=2,
                             global_iterations=3, population_size=6,
                             progress=parallel_progress.append)
        self.assertTrue(any("2 CPU workers" in line and "generation 1/3" in line
                            for line in parallel_progress))
        self.assertTrue(parallel.converged)
        self.assertGreater(parallel.simulation_count, 0)
        self.assertTrue(any("PBM" in line and "simulation(s) in" in line for line in parallel_progress))
        self.assertIsNotNone(parallel.intervals, parallel.uncertainty_error)
        np.testing.assert_allclose(parallel.parameters, [6], atol=0.01)
        self.assertLess(parallel.ssr, 1e-9)

    def test_completed_fit_is_retained_when_covariance_is_unidentifiable(self):
        class ConstantObjective:
            residual_count = sample_count = 4
            def residuals(self, parameters, *, global_stage=False):
                return np.ones(4)

        definitions = np.zeros((35, 4))
        definitions[0] = [1, 1., 0., 2.]
        result = fit_model({"paraset": definitions}, ConstantObjective(), global_iterations=0)
        self.assertTrue(result.converged)
        self.assertIsNone(result.intervals)
        self.assertTrue(result.uncertainty_error)
        self.assertTrue(np.isfinite(result.parameters).all())

    def test_save_model_replaces_matlab_optimizer_settings_with_python_settings(self):
        original_mesh = {"Lmax": 250, "N": 250}
        original = {"pbmsetting": original_mesh, "ExtraData": np.arange(3),
                    "Solubility": {"a": 0.1}, "gasetting": {"PopulationSize": 40}}
        settings = {"pbmsetting": {"Lmax": 900, "N": 120, "RelTol": 2e-7},
                    "pyfitsettings": {"population_size": 12, "global_tolerance": 1e-6}}
        with TemporaryDirectory() as directory:
            source, output = (Path(directory) / f"{name}.mat" for name in ("source", "output"))
            savemat(source, original, do_compression=True)
            self.assertNotIn("gasetting", read_model(source))
            save_model(source, output, {}, settings)
            saved = loadmat(output, simplify_cells=True)
            self.assertEqual(read_model(output)["pyfitsettings"], settings["pyfitsettings"])
            untouched = loadmat(source, simplify_cells=True)
            self.assertEqual(untouched["pbmsetting"], original_mesh)
            self.assertEqual(untouched["gasetting"], original["gasetting"])
        self.assertEqual(saved["pbmsetting"], settings["pbmsetting"])
        self.assertNotIn("gasetting", saved)
        np.testing.assert_array_equal(saved["ExtraData"], original["ExtraData"])
