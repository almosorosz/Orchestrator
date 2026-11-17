from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

from imed_python.analysis_budget import AnalysisBudget
from imed_python import test_mesh_validation
from imed_python.comparison import compare_model
from imed_python.mesh_validation import validate_meshes


class AnalysisBudgetTests(unittest.TestCase):
    def test_deadline_and_stop_share_a_reusable_callback(self):
        budget = AnalysisBudget()
        check = budget.check
        with patch("imed_python.analysis_budget.monotonic", side_effect=[100., 101., 103., 200., 201.]):
            budget.start(3.)
            self.assertFalse(check())
            self.assertTrue(check())
            budget.start(3.)
            self.assertIs(budget.check, check)
            self.assertFalse(check())
        self.assertEqual(budget.apply({"MaxIterationTime": 30.})["MaxIterationTime"], 3.)
        budget.stop()
        self.assertTrue(check())
        self.assertFalse(budget.expired())

    def test_mesh_validation_retains_completed_meshes_when_budget_expires(self):
        case = test_mesh_validation.MeshValidationTests.case()
        cutoff = Event()

        def progress(message):
            if "Mesh 2/3" in message:
                cutoff.set()

        result = validate_meshes(case, 0, 0, [20, 30, 40], cancelled=cutoff.is_set,
                                 budget_expired=cutoff.is_set, progress=progress)
        self.assertTrue(result["budget_expired"])
        np.testing.assert_array_equal(result["planned_bins"], [20, 30, 40])
        self.assertEqual([row["bins"] for row in result["runs"]], [20])
        self.assertGreater(result["runs"][0]["pbm_seconds"], 0.)
        with self.assertRaisesRegex(TimeoutError, "before the first mesh"):
            validate_meshes(case, 0, 0, [20, 30, 40], cancelled=cutoff.is_set, budget_expired=cutoff.is_set)

    def test_comparison_preserves_pbm_when_qmom_runs_out_of_time(self):
        case = test_mesh_validation.MeshValidationTests.case()
        cutoff = Event()

        def progress(message):
            if "QMOM reference" in message:
                cutoff.set()

        inputs = (case["ModelParameter"][0], case["ExpData"][0], case["ProcInf"][0],
                  case["Solubility"], case["pbmsetting"])
        result = compare_model(*inputs, cancelled=cutoff.is_set, budget_expired=cutoff.is_set, progress=progress)
        self.assertIn("pbm", result)
        self.assertNotIn("qmom", result)
        self.assertIn("Time budget reached", result["qmom_error"])
