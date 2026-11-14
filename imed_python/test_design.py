import unittest
from unittest.mock import patch

import numpy as np

from imed_python.design import discrimination_score, d_optimal_log_score, optimize_recipe


def recipe_score(experiment, process):
    time, _ = experiment["TemperatureData"]
    return time[1] - process["SeedMass"]


class DesignTests(unittest.TestCase):
    @staticmethod
    def model(value):
        definitions = np.zeros((35, 4))
        definitions[0] = [1, 2, 1, 3]
        return {"paraset": definitions, "para": [value], "covmat": [[0.01]]}

    @staticmethod
    def simulate_linear(parameters, *_):
        value = parameters[0]
        time = [0, 1]
        return np.array([time, [value, value]]), np.array([time, [value, value]]), None

    @patch("imed_python.design.simulate", side_effect=simulate_linear.__func__)
    def test_model_discrimination_uses_pairwise_weighted_separation(self, _):
        score = discrimination_score([self.model(1), self.model(2), self.model(3)],
                                     {}, {}, {}, {}, [0, 2, 0], experiment_count=2)
        self.assertAlmostEqual(score, 6)

    @patch("imed_python.design.simulate", side_effect=simulate_linear.__func__)
    def test_d_optimal_score_matches_one_parameter_fisher_matrix(self, _):
        score = d_optimal_log_score(self.model(2), {}, {}, {}, {}, [1, 1, 0],
                                    [[1.5], [2.5]])
        self.assertAlmostEqual(score, np.log(16))

    def test_search_preserves_added_mass_and_ramp_limits(self):
        conditions = {"batch": 30, "temperature": (20, 15), "temperature_rates": (0.5, 1),
                      "liquid_masses": (10, 0), "concentration": 0.1, "final_anti": 0.2,
                      "antisolvent_rates": (0.2, 0.5), "seed_bounds": (0, 0)}

        def score(experiment, _):
            time, temperature = experiment["TemperatureData"]
            flow_time, flow = experiment["FlowRateData"]
            self.assertTrue((np.diff(time) > 0).all())
            self.assertTrue((np.diff(flow_time) > 0).all())
            self.assertAlmostEqual(np.trapezoid(flow, flow_time), 2.5)
            ramps = np.abs(np.diff(temperature) / np.diff(time))
            self.assertTrue(((ramps[ramps > 0] >= 0.5) & (ramps <= 1)[ramps > 0]).all())
            return time[1]

        result, _, _ = optimize_recipe({}, {}, conditions, score, iterations=2)
        self.assertGreater(result, 0)

    def test_parallel_search_returns_the_recipe_it_scored(self):
        conditions = {"batch": 30, "temperature": (20, 15), "temperature_rates": (0.5, 1),
                      "liquid_masses": (10, 0), "concentration": 0.1, "final_anti": 0.2,
                      "antisolvent_rates": (0.2, 0.5), "seed_bounds": (0, 1)}
        score, experiment, process = optimize_recipe(
            {}, {}, conditions, recipe_score, iterations=2, workers=2)
        self.assertGreater(score, 10)
        self.assertEqual(score, recipe_score(experiment, process))
        time, flow = experiment["FlowRateData"]
        self.assertAlmostEqual(np.trapezoid(flow, time), 2.5)
