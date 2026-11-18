from copy import deepcopy
import unittest

import numpy as np

from imed_python.distribution import volume_distribution
from imed_python.evaluation import absolute_percentage_error, evaluate_model, response_error
from imed_python.solver import simulate
from imed_python.validation.cases import Case


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def case():
        parameters, experiment, process, solubility, settings = Case("cooling", "growth").inputs(20)
        count, concentration, csd = simulate(parameters, experiment, process, solubility, settings)
        _, sizes = volume_distribution(csd, process, settings)
        experiment.update(InitialConcentration=.1, FinalD10D50D90=sizes*1.1,
                          CountsData=np.vstack((count[0], count[1]*20)),
                          ConcentrationData=np.vstack((concentration[0], concentration[1]*1.1)))
        definitions = np.column_stack((np.zeros(35), parameters, parameters, parameters))
        return {"ModelParameter": [{"paraset": definitions}], "ExpData": [experiment],
                "ProcInf": [process], "Solubility": solubility, "pbmsetting": settings}

    def test_errors_use_measured_denominator_and_omit_zero_observations(self):
        np.testing.assert_allclose(absolute_percentage_error([10., 0., 20.], [12., 1., 18.]),
                                   [20., np.nan, 10.], equal_nan=True)
        self.assertAlmostEqual(response_error([[0., 1.], [0., 2.]], np.array([[0., 1.], [2., 1.]])), 50.)

    def test_evaluation_matches_known_size_concentration_and_count_errors(self):
        result = evaluate_model(self.case(), 0)
        record = result["records"][0]
        self.assertEqual(record["error"], "Success")
        np.testing.assert_allclose(record["size_errors"], 100/11)
        self.assertAlmostEqual(record["concentration_mape"], 100/11)
        self.assertAlmostEqual(record["count_shape_mape"], 0., places=10)
        self.assertGreater(record["seconds"], 0.)

    def test_failed_experiments_retain_their_indices_and_successful_results(self):
        case = self.case()
        invalid = deepcopy(case["ExpData"][0])
        invalid["InitialConcentration"] = 2.
        case["ExpData"].append(invalid)
        case["ProcInf"].append(case["ProcInf"][0])
        result = evaluate_model(case, 0)
        self.assertEqual(result["records"][0]["error"], "Success")
        self.assertNotEqual(result["records"][1]["error"], "Success")
        self.assertEqual(result["records"][1]["experiment_index"], 1)
        with self.assertRaises(InterruptedError):
            evaluate_model(case, 0, cancelled=lambda: True)
