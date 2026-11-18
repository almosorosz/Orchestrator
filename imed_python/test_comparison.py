import unittest

import numpy as np

from imed_python.comparison import compare_model
from imed_python.validation.cases import Case


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parameters, experiment, process, solubility, settings = Case("combined", "full").inputs(80)
        cls.model = {"paraset": np.column_stack((np.zeros(35), parameters, parameters, parameters))}
        cls.experiment, cls.process, cls.solubility, cls.settings = experiment, process, solubility, settings
        cls.result = compare_model(cls.model, experiment, process, solubility, settings)

    def test_comparison_preserves_inventory_and_volume_normalization(self):
        result = self.result
        self.assertGreater(result["pbm_seconds"], 0)
        self.assertGreater(result["qmom_seconds"], 0)
        self.assertGreaterEqual(result["total_seconds"], result["pbm_seconds"] + result["qmom_seconds"])
        self.assertIn("qmom", result, result.get("qmom_error"))
        for method in ("pbm", "qmom"):
            values = result[method]
            total = values["dissolved"] + values["crystal"] + values["overflow"]
            np.testing.assert_allclose(total, result["initial_api"], rtol=1e-7, atol=1e-10)
            self.assertAlmostEqual(np.sum(values["final_density"]*np.diff(result["bounds"])), 100.)
            self.assertEqual(values["moments"].shape, (6, 100))
            self.assertEqual(values["percentiles"].shape, (3, 100))
            self.assertTrue(np.all(np.diff(values["percentiles"], axis=0) >= 0))

    def test_stop_interrupts_comparison_before_simulating(self):
        with self.assertRaises(InterruptedError):
            compare_model(self.model, self.experiment, self.process, self.solubility,
                          self.settings, cancelled=lambda: True)
