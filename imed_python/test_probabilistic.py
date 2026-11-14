from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from imed_python.model_file import save_checkpoint
from imed_python.probabilistic import (
    ERROR_LEVELS, _trial, assess_reliability, error_limits, fbrm_calibration,
    joint_parameter_draws, model_intervals, partial_rank_correlations,
    prediction_errors, prcc_diagnostics, propagate, sample_intervals,
)
from imed_python.session import history, pack_intervals, record_step, signature
from imed_python.solver import load_case
from imed_python.uncertainty import intervals_from_covariance
from imed_python.validation.cases import Case


class ProbabilisticTests(unittest.TestCase):
    @staticmethod
    def case():
        parameters, experiment, process, solubility, settings = Case("combined", "growth").inputs(20)
        table = np.column_stack((np.zeros(35), parameters, parameters, parameters))
        table[12] = [1, parameters[12], -2, 1]
        case = {"ModelParameter": [{"paraset": table}], "ExpData": [experiment], "ProcInf": [process],
                "Solubility": solubility, "pbmsetting": settings}
        interval = intervals_from_covariance([12], parameters[[12]], [[.0001]], 10, source="Current nominal values")
        record_step(case, "parameter_uncertainty", {"intervals": pack_intervals(interval)},
                    model_index=0, context=signature(case, 0, include_weights=False))
        return case

    def test_joint_draws_reproduce_covariance_and_reject_indefinite_matrix(self):
        covariance = np.array([[.04, -.015], [-.015, .01]])
        interval = intervals_from_covariance([5, 6], [2, 1], covariance, 40)
        draws = joint_parameter_draws(interval, 150000, np.random.default_rng(0))
        np.testing.assert_allclose(np.cov(draws.T), covariance, rtol=.015)
        np.testing.assert_allclose(draws.mean(axis=0), interval.values, atol=.002)
        invalid = intervals_from_covariance([5, 6], [2, 1], [[1, 2], [2, 1]], 40)
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            joint_parameter_draws(invalid, 10, np.random.default_rng(0))

    def test_error_limits_match_matlab_midpoint_percentiles_of_absolute_errors(self):
        np.testing.assert_allclose(error_limits([-8, -2, 1, 4], [25, 50, 75]), [1.5, 3, 6])

    def test_condition_draws_modify_selected_profiles_without_mutating_case(self):
        case = self.case()
        experiment, process = deepcopy(case["ExpData"][0]), deepcopy(case["ProcInf"][0])
        with patch("imed_python.probabilistic._predict", return_value={}) as predict:
            _trial([3., 1.2, .8, 1.1, 1.5], baseline=case["ModelParameter"][0]["paraset"][:, 1],
                   indices=[], condition_keys=["temperature_offset", "feed_factor", "seed_factor",
                                               "concentration_factor", "duration_factor"],
                   experiment=experiment, process=process, solubility=case["Solubility"], settings=case["pbmsetting"])
        modified, info = predict.call_args.args[1:3]
        np.testing.assert_allclose(modified["TemperatureData"][0], np.asarray(experiment["TemperatureData"])[0]*1.5)
        np.testing.assert_allclose(modified["TemperatureData"][1], np.asarray(experiment["TemperatureData"])[1]+3)
        np.testing.assert_allclose(modified["FlowRateData"][1], np.asarray(experiment["FlowRateData"])[1]*1.2)
        self.assertAlmostEqual(info["SeedMass"], process["SeedMass"]*.8)
        self.assertAlmostEqual(modified["InitialConcentration"], .11)
        self.assertNotIn("InitialConcentration", experiment)

    def test_propagation_retains_draws_failures_and_checkpoint_covariance(self):
        case = self.case()
        result = propagate(case, 0, 0, samples=4)
        self.assertTrue(result["valid"].all())
        self.assertEqual(result["predictions"].shape, (4, 3, 100))
        self.assertEqual(result["counts"].shape, (4, 100))
        self.assertEqual(result["prcc"].shape, (1, 5))
        record_step(case, "model_reliability", result, model_index=0,
                    context=signature(case, 0, include_weights=False))
        with TemporaryDirectory() as directory:
            file = Path(directory)/"sampled.mat"
            save_checkpoint(None, file, case)
            restored = load_case(file)
        np.testing.assert_array_equal(model_intervals(restored, 0).covariance, [[.0001]])
        saved = history(restored)[-1]["result"]
        np.testing.assert_array_equal(saved["inputs"].reshape(4, 1), result["inputs"])
        np.testing.assert_array_equal(saved["predictions"], result["predictions"])
        np.testing.assert_array_equal(saved["counts"], result["counts"])
        self.assertEqual(len(saved["failure_messages"]), 4)
        self.assertIsInstance(saved["observed"], dict)

    def test_conditions_can_run_without_parameter_ci_and_stop_is_propagated(self):
        case = self.case()
        case["PythonWorkflow"]["steps"] = []
        result = propagate(case, 0, 0, samples=5, use_parameters=False,
                           conditions={"temperature_offset": [-1, 1]})
        self.assertEqual(result["inputs"].shape, (5, 1))
        self.assertTrue(result["valid"].all())
        with self.assertRaisesRegex(ValueError, "Calculate parameter CIs"):
            propagate(case, 0, 0, samples=2)
        with self.assertRaises(InterruptedError):
            propagate(self.case(), 0, 0, samples=2, cancelled=lambda: True)

    def test_prcc_distinguishes_input_effects_and_handles_constant_outputs(self):
        rng = np.random.default_rng(3)
        inputs = rng.normal(size=(300, 2))
        outputs = np.column_stack((inputs[:, 0], -inputs[:, 1], np.ones(300)))
        result = partial_rank_correlations(inputs, outputs)
        self.assertAlmostEqual(result[0, 0], 1.)
        self.assertAlmostEqual(result[1, 1], -1.)
        self.assertTrue(np.isnan(result[:, 2]).all())
        self.assertTrue(np.isnan(partial_rank_correlations(inputs[:2], outputs[:2])).all())

    def test_missing_size_predictions_do_not_hide_valid_concentration_prcc(self):
        inputs = np.arange(10.)[:, None]
        outputs = np.column_stack((np.full(10, np.nan), inputs[:, 0]))
        result = partial_rank_correlations(inputs, outputs)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertAlmostEqual(result[0, 1], 1.)
        diagnostics = prcc_diagnostics(inputs, outputs)
        np.testing.assert_array_equal(diagnostics["draw_counts"], [0, 10])
        self.assertEqual(diagnostics["minimum_draws"], 4)


class ReliabilityAssessmentTests(unittest.TestCase):
    @staticmethod
    def result():
        predictions = np.full((2, 3, 2), 10.)
        predictions[1, 0, 0] = 20.
        nominal = {"percentiles": np.full((3, 2), 10.)}
        observed = {name: np.array([10., 10.]) for name in ("D10", "D50", "D90")}
        times = {name: np.array([0., 1.]) for name in observed}
        return {"time": [0., 1.], "predictions": predictions, "nominal": nominal,
                "valid": [True, True, False], "observed": observed, "observation_time": times}

    def test_empirical_reliability_retains_failed_draws_and_uses_matlab_denominator(self):
        errors = prediction_errors(self.result())
        self.assertEqual(errors["D10"]["relative"][1, 0], 50.)
        self.assertTrue(np.isnan(errors["D10"]["relative"][2]).all())
        assessment = assess_reliability(errors, 10.)
        d10 = assessment["responses"]["D10"]
        self.assertEqual(d10["comparison_successes"], 3)
        self.assertEqual(d10["comparison_count"], 6)
        self.assertEqual(d10["comparison_reliability"], 50.)
        self.assertEqual(d10["passing_draw_count"], 1)
        self.assertAlmostEqual(assessment["joint_reliability"], 100/3)

    def test_final_scope_and_updated_threshold_reuse_the_same_predictions(self):
        result = self.result()
        final = assess_reliability(prediction_errors(result, scope="final"), 10.)
        self.assertAlmostEqual(final["joint_reliability"], 200/3)
        self.assertEqual(final["responses"]["D10"]["comparison_count"], 3)
        all_points = assess_reliability(prediction_errors(result), 50.)
        self.assertAlmostEqual(all_points["joint_reliability"], 200/3)
        self.assertEqual(all_points["responses"]["D10"]["passing_draw_count"], 2)

    def test_zero_predictions_and_missing_measurements_have_explicit_outcomes(self):
        result = self.result()
        result["predictions"][0, 0, 0] = 0.
        errors = prediction_errors(result)
        self.assertTrue(np.isnan(errors["D10"]["relative"][0, 0]))
        self.assertEqual(assess_reliability(errors, 10.)["responses"]["D10"]["passing_draw_count"], 0)
        result["observed"]["D90"] = np.empty(0)
        result["observation_time"]["D90"] = np.empty(0)
        assessment = assess_reliability(prediction_errors(result), 10.)
        self.assertFalse(assessment["joint_available"])
        self.assertTrue(np.isnan(assessment["joint_reliability"]))
        self.assertEqual(assessment["responses"]["D90"]["comparison_count"], 0)

    def test_ten_passing_draws_report_finite_sampling_uncertainty(self):
        result = self.result()
        result["predictions"] = np.full((10, 3, 2), 10.)
        result["valid"] = np.ones(10, dtype=bool)
        assessment = assess_reliability(prediction_errors(result), 5.)
        self.assertEqual(assessment["joint_reliability"], 100.)
        np.testing.assert_allclose(assessment["joint_sampling_interval"], [69.1502892181239, 100.])
        self.assertEqual(ERROR_LEVELS, (95., 99.))

    def test_fbrm_scaling_is_fixed_from_the_nominal_trace(self):
        nominal = {"time": np.array([0., 1., 2.]), "counts": np.array([2., 4., 6.])}
        measured = {"CountsData": [[0., 1., 2.], [50., 100., 150.]]}
        calibration = fbrm_calibration(nominal, measured)
        self.assertTrue(calibration["scaled"])
        self.assertEqual(calibration["scale"], 25.)
        self.assertFalse(fbrm_calibration(nominal, {})["scaled"])
        nominal["counts"][:] = 0
        self.assertFalse(fbrm_calibration(nominal, measured)["scaled"])

    def test_model_intervals_use_central_percentiles_and_retain_undefined_sizes(self):
        values = np.column_stack((np.arange(100.), np.full(100, np.nan)))
        intervals = sample_intervals(values)
        self.assertEqual(intervals["median"][0], 49.5)
        self.assertEqual(intervals["lower95"][0], 2.)
        self.assertEqual(intervals["upper95"][0], 97.)
        self.assertEqual(intervals["lower99"][0], 0.)
        self.assertEqual(intervals["upper99"][0], 99.)
        self.assertTrue(all(np.isnan(value[1]) for value in intervals.values()))
        self.assertTrue(all(np.isnan(value).all() for value in sample_intervals(np.full((2, 4), np.nan)).values()))
