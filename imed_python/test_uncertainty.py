import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm, t

from imed_python.estimation import FitObjective
from imed_python.uncertainty import intervals_from_covariance, local_intervals


class LinearResiduals:
    cancelled = None
    sample_count = 5

    def __init__(self, matrix):
        self.matrix = matrix
        self.noise = np.array([0.01, -0.02, 0.01, 0., 0.])

    def residuals(self, parameters):
        if parameters[0] < 1:
            raise AssertionError("A sensitivity trial crossed the parameter bound.")
        return self.matrix @ (parameters[:2] - [1, 2]) - self.noise


class UncertaintyTests(unittest.TestCase):
    @staticmethod
    def model():
        definitions = np.zeros((35, 4))
        definitions[0] = [1, 1, 1, 3]
        definitions[1] = [1, 2, 0, 4]
        definitions[2] = [1, 0, 0, 0]  # Fixed by equal bounds.
        definitions[8] = [1, 1, 0, 1]  # Discrete switch.
        return {"paraset": definitions}

    def test_nominal_values_give_local_t_intervals_without_a_fit(self):
        model = self.model()
        matrix = np.column_stack((np.ones(5), np.arange(5)))
        objective = LinearResiduals(matrix)
        expected_covariance = (objective.noise @ objective.noise / 3
                               * np.linalg.inv(matrix.T @ matrix))
        result = local_intervals(model, objective, convention="student_t")
        self.assertEqual(result.source, "Nominal values")
        self.assertEqual(result.degrees_of_freedom, 3)
        np.testing.assert_array_equal(result.indices, [0, 1])
        np.testing.assert_array_equal(result.values, [1, 2])
        np.testing.assert_allclose(result.covariance, expected_covariance)
        width = t.ppf(0.975, 3) * np.sqrt(expected_covariance.diagonal())
        np.testing.assert_allclose(result.lower, result.values - width)
        np.testing.assert_allclose(result.upper, result.values + width)
        self.assertNotIn("para", model)

    def test_existing_fit_is_used_as_the_interval_center(self):
        model = {**self.model(), "para": [1.01, 1.98, 0., 1.]}
        objective = LinearResiduals(np.column_stack((np.ones(5), np.arange(5))))
        result = local_intervals(model, objective, confidence=0.90)
        self.assertEqual(result.source, "Fitted values")
        self.assertEqual(result.confidence, 0.90)
        np.testing.assert_array_equal(result.values, model["para"][:2])

    def test_matlab_interval_uses_normal_multiplier(self):
        covariance = np.array([[4., -1.], [-1., 9.]])
        result = intervals_from_covariance([0, 1], [2., 3.], covariance, 3)
        self.assertEqual(result.convention, "matlab")
        np.testing.assert_allclose(result.upper, [2., 3.] + norm.ppf(.975)*np.array([2., 3.]))
        np.testing.assert_allclose(result.lower, [2., 3.] - norm.ppf(.975)*np.array([2., 3.]))

    def test_unidentifiable_parameters_do_not_report_confidence_limits(self):
        objective = LinearResiduals(np.ones((5, 2)))
        with self.assertRaisesRegex(ValueError, "cannot distinguish"):
            local_intervals(self.model(), objective)
        objective.sample_count = 2
        with self.assertRaisesRegex(ValueError, "More independent measurements"):
            local_intervals(self.model(), objective)

    def test_residuals_use_measured_sizes_and_exclude_initial_concentration(self):
        experiment = {"FinalCSDsData": [[0., 1.5, 2.5, 3.5, 4.], [0., 1., 2., 1., 0.]],
                      "ConcentrationData": [[0., 1., 2.], [0.1, 0.09, 0.08]]}
        objective = FitObjective([experiment], [{"ShapeFactor": 1}], {},
                                 {"Lmax": 4, "N": 4}, [0, 1, 1], "CSD", at_measurements=True)
        self.assertEqual(objective.residual_count, 5)
        self.assertEqual(objective.sample_count, 4)
        outputs = (None, np.array([[0., 1., 2.], [0.1, 0.088, 0.081]]),
                   np.array([[0.5, 1.5, 2.5, 3.5], [1., 1., 1., 1.]]))
        with patch("imed_python.estimation.simulate", return_value=outputs):
            residuals = objective.residuals(np.zeros(35))
        np.testing.assert_allclose(residuals[:2], [-0.002 / np.sqrt(2), 0.001 / np.sqrt(2)])
        self.assertAlmostEqual(residuals[2:].sum(), 0.)
