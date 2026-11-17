import unittest
from unittest.mock import patch

import numpy as np

from imed_python.mesh_validation import mesh_counts, validate_meshes
from imed_python.validation.cases import Case


class MeshValidationTests(unittest.TestCase):
    @staticmethod
    def case():
        parameters, experiment, process, solubility, settings = Case("cooling", "growth").inputs(30)
        model = {"paraset": np.column_stack((np.zeros(35), parameters, parameters, parameters))}
        return {"ModelParameter": [model], "ExpData": [experiment], "ProcInf": [process],
                "Solubility": solubility, "pbmsetting": settings}

    def test_counts_include_endpoints_and_are_distinct(self):
        np.testing.assert_array_equal(mesh_counts(100, 1000, 3), [100, 550, 1000])
        np.testing.assert_array_equal(mesh_counts(100, 104, 4), [100, 101, 103, 104])
        with self.assertRaises(ValueError):
            mesh_counts(100, 101, 3)
        with self.assertRaises(ValueError):
            mesh_counts(1000, 100, 3)

    def test_scipy_reports_volume_distribution_and_conservative_inventory(self):
        result = validate_meshes(self.case(), 0, 0, [20, 30, 40])
        self.assertFalse(result["uses_jax"])
        for row in result["runs"]:
            self.assertEqual(row["status"], "Complete")
            density = row["density"]
            self.assertEqual(len(density), row["bins"])
            self.assertAlmostEqual(density.sum() * 65 / row["bins"], 100)
            self.assertTrue(np.all(np.diff(row["percentiles"]) > 0))
            self.assertGreater(row["pbm_seconds"], 0)
            self.assertLess(row["mass_error_ppm"], .01)

    def test_jax_warms_each_mesh_and_uses_second_solve(self):
        case = self.case()
        case["pbmsetting"]["Numerics"] = "jax"
        counts = [20, 30, 40]
        from imed_python.solver import simulate

        def outputs(bins):
            mesh = {**case["pbmsetting"], "Numerics": "conservative", "N": bins}
            return simulate(case["ModelParameter"][0]["paraset"][:, 1], case["ExpData"][0], case["ProcInf"][0],
                            case["Solubility"], mesh, return_history=True)

        # Warm-ups have no usable outputs so only the timed runs can populate the plots.
        solutions = []
        for bins in counts:
            solutions.extend([None, outputs(bins)])
        with (patch("imed_python.mesh_validation.pbm_backend_label", return_value="JAX PBM (test)"),
              patch("imed_python.mesh_validation.simulate", side_effect=solutions) as solve,
              patch("imed_python.mesh_validation.monotonic", side_effect=[10, 11, 20, 22, 30, 33])):
            result = validate_meshes(case, 0, 0, counts)
        self.assertTrue(result["uses_jax"])
        self.assertEqual([call.args[-1]["N"] for call in solve.call_args_list], [20, 20, 30, 30, 40, 40])
        self.assertEqual([row["pbm_seconds"] for row in result["runs"]], [1., 2., 3.])
        self.assertTrue(all(row["status"] == "Complete" and np.size(row["density"]) for row in result["runs"]))
