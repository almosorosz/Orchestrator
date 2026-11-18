from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.io import loadmat, savemat

from imed_python.estimation import FitResult
from imed_python.model_file import save_checkpoint
from imed_python.session import (
    history, latest_step, pack_fit, pack_intervals, record_step, signature, unpack_fit, unpack_intervals,
)
from imed_python.solver import load_case
from imed_python.uncertainty import intervals_from_covariance


class SessionTests(unittest.TestCase):
    @staticmethod
    def case():
        parameters = np.zeros((35, 4))
        parameters[0] = [1, 2, 1, 3]
        return {"ModelParameter": [{"paraset": parameters}, {"paraset": parameters.copy(), "para": [2.]}],
                "ExpData": [{"TemperatureData": [[0, 10], [20, 20]],
                             "ConcentrationData": [[0], [0.1]]}],
                "ProcInf": [{"ShapeFactor": 1.}], "Solubility": {"a": 0.},
                "pbmsetting": {"N": 10, "Numerics": "conservative"},
                "objweightfactor": {"W1": 0., "W2": 1., "W3": 0., "Terms": "CSD"}}

    def test_checkpoint_retains_results_and_signatures_after_mat_roundtrip(self):
        case = self.case()
        case["ExpData"].append({**case["ExpData"][0], "FinalD10D50D90": [10., 20., 30.]})
        case["ProcInf"].append(case["ProcInf"][0].copy())
        interval = intervals_from_covariance([0], [2.], [[0.04]], 4)
        record_step(case, "parameter_uncertainty", {"intervals": pack_intervals(interval)},
                    model_index=0, context=signature(case, 0, include_weights=False))
        fit = FitResult(np.array([2.]), np.arange(5.), np.ones((5, 1)), np.array([[0.04]]),
                        30., 0., True, interval)
        record_step(case, "parameter_estimation", pack_fit(fit), model_index=1,
                    context=signature(case, 1, include_values=False))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.mat"
            save_checkpoint(None, path, case)
            restored = load_case(path)
            self.assertIsNotNone(loadmat(path)["ModelParameter"].dtype.names)
        self.assertEqual(len(history(restored)), 2)
        step = latest_step(restored, "parameter_uncertainty", model_index=0,
                           context=signature(restored, 0, include_weights=False))
        self.assertIsNotNone(step)
        loaded = unpack_intervals(step["result"]["intervals"])
        np.testing.assert_array_equal(loaded.covariance, interval.covariance)
        np.testing.assert_array_equal(loaded.lower, interval.lower)
        step = latest_step(restored, "parameter_estimation", model_index=1,
                           context=signature(restored, 1, include_values=False))
        self.assertIsNotNone(step)
        loaded_fit = unpack_fit(step["result"])
        np.testing.assert_array_equal(loaded_fit.intervals.upper, interval.upper)
        self.assertEqual(loaded_fit.jacobian.shape, (5, 1))
        restored["ModelParameter"][0]["paraset"][0, 1] = 2.1
        self.assertIsNone(latest_step(restored, "parameter_uncertainty", model_index=0,
                                     context=signature(restored, 0, include_weights=False)))

    def test_source_checkpoint_preserves_legacy_variables_and_can_be_updated_atomically(self):
        case = self.case()
        with TemporaryDirectory() as directory:
            source, saved = (Path(directory) / name for name in ("source.mat", "saved.mat"))
            savemat(source, {"ExtraData": np.arange(4), "gasetting": {"PopulationSize": 200}})
            record_step(case, "simulation", {"count": np.array([[0., 1.], [3., 4.]])})
            save_checkpoint(source, saved, case)
            record_step(case, "model_configuration", {"paraset": case["ModelParameter"][0]["paraset"]},
                        model_index=0)
            save_checkpoint(saved, saved, case)
            restored = load_case(saved)
            raw = loadmat(saved, simplify_cells=True)
        self.assertEqual(len(history(restored)), 2)
        self.assertIn("gasetting", raw)
        np.testing.assert_array_equal(raw["ExtraData"], np.arange(4))
