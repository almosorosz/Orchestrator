from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from imed_python.model_file import save_checkpoint
from imed_python.probabilistic import model_intervals, propagate
from imed_python.session import history, pack_intervals, record_step, signature
from imed_python.simulated_profile import editable_groups, prediction_inputs
from imed_python.solver import load_case
from imed_python.uncertainty import intervals_from_covariance
from imed_python.validation.cases import Case


class SimulatedProfileTests(unittest.TestCase):
    @staticmethod
    def case():
        parameters, experiment, process, solubility, settings = Case("combined", "growth").inputs(20)
        table = np.column_stack((np.zeros(len(parameters)), parameters, parameters, parameters))
        table[12] = [1, parameters[12], -2, 1]
        return {"ModelParameter": [{"paraset": table}], "ExpData": [experiment], "ProcInf": [process],
                "Solubility": solubility, "pbmsetting": settings}

    def test_doe_factors_and_selected_kinetics_gate_editable_inputs(self):
        case = self.case()
        case["PythonWorkflow"] = {"doe_factors": ["AS_am", "stirring_rate"]}
        self.assertEqual(editable_groups(case, 0), {"feed"})
        parameters = case["ModelParameter"][0]["paraset"]
        parameters[5, 1], parameters[11, 1] = -3, 1
        self.assertEqual(editable_groups(case, 0), {"feed", "stirring"})
        case["PythonWorkflow"]["doe_factors"] = ["stirring_rate_seed", "t_cool", "seed"]
        self.assertEqual(editable_groups(case, 0), {"temperature", "seed"})

    def test_legacy_profiles_compare_programs_across_experiments(self):
        case = self.case()
        experiment = case["ExpData"][0]
        experiment.update(TemperatureData=[[0, 10], [40, 30]], FlowRateData=[[0, 10], [0, 0]])
        second = {**experiment, "TemperatureData": [[0, 5, 10], [40, 35, 30]]}
        case["ExpData"].append(second)
        case["ProcInf"].append(dict(case["ProcInf"][0]))
        self.assertEqual(editable_groups(case, 0), set())
        second["TemperatureData"][1][-1] = 25
        self.assertEqual(editable_groups(case, 0), {"temperature"})

    def test_profile_builds_new_inputs_with_seed_and_power(self):
        case = self.case()
        template, template_process = prediction_inputs(case, 0)
        original = deepcopy(template_process)
        template["InitialCSDsData"] = []
        template["D10Data"] = np.array([[0, 10], [5, 7]])
        recipe = {"temperature": [[0, 20, 40], [50, 35, 30]], "feed": [[0, 10, 30], [.2, .2, 0]],
                  "seed_mass": .02, "seed_percentiles": [10, 20, 40], "initial_antisolvent": 3,
                  "stirring_rpm": 400, "reference_rpm": 200}
        experiment, process = prediction_inputs(case, 0, recipe)
        np.testing.assert_equal(experiment["TemperatureData"], recipe["temperature"])
        np.testing.assert_equal(experiment["FlowRateData"][:, -1], [40, 0])
        self.assertGreater(np.size(experiment["InitialCSDsData"]), 0)
        self.assertEqual(process["SeedMass"], .02)
        self.assertEqual(process["AntiSolvMass"], 3)
        self.assertEqual(process["PowerRatio"], original["PowerRatio"] * 8)
        self.assertEqual(template_process, original)
        self.assertNotIn("D10Data", experiment)
        with self.assertRaisesRegex(ValueError, "seed D10"):
            prediction_inputs(case, 0, {"seed_mass": .02})
        with self.assertRaisesRegex(ValueError, "increasing"):
            prediction_inputs(case, 0, {"temperature": [[0, 2, 2], [50, 35, 30]]})

    def test_profile_runs_with_training_covariance_and_round_trips(self):
        case = self.case()
        context = signature(case, 0, include_weights=False)
        value = case["ModelParameter"][0]["paraset"][12, 1]
        intervals = intervals_from_covariance([12], [value], [[.0001]], 10)
        record_step(case, "parameter_uncertainty", {"intervals": pack_intervals(intervals)},
                    model_index=0, context=context)
        case["PythonWorkflow"]["doe_factors"] = np.array(["T_final", "AS_am"], dtype=object)
        recipe = {"name": "New cooling profile", "temperature": [[0, 120], [30, 20]],
                  "feed": [[0, 60, 120], [0., .01, 0.]]}
        result = propagate(case, 0, 0, samples=4, simulated_profile=recipe)
        self.assertTrue(result["valid"].all())
        self.assertAlmostEqual(result["time"][-1], 120)
        self.assertFalse(result["fbrm"]["scaled"])
        self.assertEqual(signature(case, 0, include_weights=False), context)
        record_step(case, "probabilistic_sensitivity", result, model_index=0, context=context)
        with TemporaryDirectory() as directory:
            file = Path(directory) / "new-profile.mat"
            save_checkpoint(None, file, case)
            restored = load_case(file)
        saved = history(restored)[-1]["result"]
        np.testing.assert_equal(saved["request"]["simulated_profile"]["temperature"], recipe["temperature"])
        np.testing.assert_equal(saved["predictions"], result["predictions"])
        np.testing.assert_equal(model_intervals(restored, 0).covariance, [[.0001]])
        self.assertEqual(editable_groups(restored, 0), {"temperature", "feed"})
