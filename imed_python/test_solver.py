import unittest
import warnings
from types import SimpleNamespace
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from imed_python.distribution import cell_volumes
from imed_python.solver import load_case, model_parameters, simulate, transport
from imed_python.validation.cases import Case
from imed_python.validation.matlab_ode45 import integrate_matlab_ode45


class ForwardModelTests(unittest.TestCase):
    def test_scipy_cleans_only_negligible_negative_tail_bins(self):
        inputs = Case("cooling", "growth").inputs(4)
        parameters, experiment, process, solubility, settings = inputs
        process = {**process, "SeedMass": 0}
        settings = {**settings, "AbsTol": 1e-5}
        for negative in (-2.3e-5, -0.1):
            def solution(_rhs, _span, initial, *, t_eval, **kwargs):
                state = np.tile(initial[:, None], (1, len(t_eval)))
                state[0] = negative
                state[1] = 2e8
                return SimpleNamespace(success=True, y=state)

            with self.subTest(negative=negative), patch("imed_python.solver.solve_ivp", solution):
                _, _, csd, history = simulate(parameters, experiment, process, solubility,
                                              settings, return_history=True)
                self.assertEqual(csd[1, 0], 0. if negative == -2.3e-5 else negative)
                self.assertEqual(history[0, -1], negative)

    def test_unstable_aggregation_fails_without_warning_cascade(self):
        parameters, experiment, process, solubility, settings = Case("combined", "full").inputs(20)
        parameters[24] = 300  # Finite coefficient, but the collision rate overflows.
        for mode in ("conservative", "matlab_pe"):
            with self.subTest(mode=mode), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaisesRegex(FloatingPointError, "PBM numerical failure: overflow"):
                    simulate(parameters, experiment, process, solubility,
                             {**settings, "Numerics": mode})

    def test_stop_interrupts_active_numpy_integrators(self):
        parameters, experiment, process, solubility, settings = Case("combined", "full").inputs(20)
        for mode in ("conservative", "matlab_pe"):
            with self.subTest(mode=mode):
                polls = iter([False] * 5 + [True])
                with self.assertRaisesRegex(InterruptedError, "PBM simulation stopped"):
                    simulate(parameters, experiment, process, solubility,
                             {**settings, "Numerics": mode}, cancelled=lambda: next(polls))

    @unittest.skipUnless(find_spec("jax") and find_spec("diffrax"), "JAX and Diffrax are optional")
    def test_stop_interrupts_active_jax_integrator(self):
        parameters, experiment, process, solubility, settings = Case("combined", "full").inputs(20)
        polls = iter([False] * 5 + [True])
        with self.assertRaisesRegex(InterruptedError, "PBM simulation stopped"):
            simulate(parameters, experiment, process, solubility,
                     {**settings, "Numerics": "jax"}, cancelled=lambda: next(polls))

    @unittest.skipUnless(find_spec("jax") and find_spec("diffrax"), "JAX and Diffrax are optional")
    def test_matlab_case_runs_on_jax_and_scipy(self):
        n = 60
        parameters, experiment, process, solubility, settings = Case("combined", "full").inputs(n)
        parameter_table = np.zeros((35, 4))
        parameter_table[:, 1] = parameters
        parameter_table[12, :2] = [1, 0.]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.mat"
            savemat(path, {"ExpData": experiment, "ProcInf": process, "Solubility": solubility,
                           "ModelParameter": {"paraset": parameter_table, "para": [parameters[12]]},
                           "pbmsetting": settings})
            case = load_case(path)

        parameters = model_parameters(case["ModelParameter"][0])
        process = case["ProcInf"][0]
        settings = case["pbmsetting"]
        inputs = parameters, case["ExpData"][0], process, case["Solubility"]
        scipy_count, scipy_c, scipy_csd = simulate(*inputs, settings)
        jax_count, jax_c, jax_csd, history = simulate(
            *inputs, {**settings, "Numerics": "jax", "JaxDevice": "cpu"}, return_history=True)
        np.testing.assert_allclose(jax_count, scipy_count, rtol=1e-3)
        np.testing.assert_allclose(jax_c, scipy_c, rtol=1e-4)
        np.testing.assert_allclose(jax_csd[1], scipy_csd[1], rtol=0,
                                   atol=1e-3 * scipy_csd[1].max())
        self.assertGreater(history[n + 4, -1], 0.)
        edges = np.linspace(settings["Lmin"], settings["Lmax"], n + 1)
        resolved = process["SolidDensity"] * (cell_volumes(edges, process["ShapeFactor"])
                                              * np.diff(edges)) @ history[:n] / 1e12
        inventory = resolved + history[n] * history[n + 1] + history[n + 5]
        np.testing.assert_allclose(inventory, inventory[0], rtol=0, atol=1e-10)

    @unittest.skipUnless(find_spec("jax") and find_spec("diffrax"), "JAX and Diffrax are optional")
    def test_requested_gpu_fails_if_the_backend_is_unavailable(self):
        from imed_python.solver import pbm_backend_label

        with patch("jax.devices", side_effect=RuntimeError("no GPU backend")) as devices:
            with self.assertRaisesRegex(RuntimeError, "could not initialize a GPU"):
                pbm_backend_label({"Numerics": "jax", "JaxDevice": "gpu"})
        devices.assert_called_once_with("gpu")

    def test_matlab_style_integrator_populates_dense_output(self):
        time = np.linspace(0, 2, 71)
        result = integrate_matlab_ode45(lambda t, y: np.array([3 * t**2, -y[1]]),
                                       time, [0., 1.], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(result[0], time**3, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(result[1], np.exp(-time), rtol=5e-8, atol=1e-10)

    def test_transport_conserves_count_with_no_boundary_outflow(self):
        density = np.array([1., 4., 2., 0.])
        width = np.array([1., 2., 3., 4.])
        derivative = transport(density, np.ones(4), width, incoming=3.)
        self.assertAlmostEqual(width @ derivative, 3.)

    def test_transport_uses_cell_density_when_both_adjusted_slopes_are_zero(self):
        density = np.array([2e-10, 1e-10])
        with np.errstate(divide="raise", invalid="raise"):
            derivative, outflow = transport(density, np.ones(2), np.ones(2), return_outflow=True)
        np.testing.assert_array_equal(derivative, [-2e-10, 1e-10])
        self.assertEqual(outflow, 1e-10)

    def test_fitted_parameters_replace_only_selected_values(self):
        table = np.zeros((35, 4))
        table[:, 1] = np.arange(35)
        table[[1, 20], 0] = 1
        expected = table[:, 1].copy()
        expected[[1, 20]] = [51, 52]
        np.testing.assert_array_equal(
            model_parameters({"paraset": table, "para": [51, 52]}), expected)

    def test_constant_antisolvent_feed_dilutes_inert_solution(self):
        parameters = np.zeros(35)
        parameters[[0, 5, 12, 18, 24, 28]] = -np.inf
        experiment = {
            "TemperatureData": [[0, 5], [20, 20]],
            "FlowRateData": [[0, 5], [2, 2]],
            "ConcentrationData": [[0], [0.1]],
        }
        process = {
            "ShapeFactor": 1, "SolidDensity": 1.4, "SolventDensity": 0.86,
            "AntiSolventDensity": 1, "PowerRatio": 1, "SeedMass": 0,
            "SolvMass": 10, "AntiSolvMass": 0,
        }
        count, concentration, csd = simulate(
            parameters, experiment, process, dict.fromkeys("abcdefgh", 0), {"N": 10})
        solute = 10 * 0.1 / 0.9
        np.testing.assert_allclose(concentration[1], solute / (solute + 10 + 2 * count[0]),
                                   rtol=1e-13)
        np.testing.assert_array_equal(count[1], 0)
        np.testing.assert_array_equal(csd[1], 0)

    def test_growth_outflow_remains_in_api_inventory(self):
        parameters = np.zeros(35)
        parameters[[0, 5, 18, 24, 28]] = -np.inf
        parameters[16] = -np.inf
        parameters[13] = 1
        experiment = {"TemperatureData": [[0, 30], [20, 20]],
                      "ConcentrationData": [[0], [0.1]],
                      "InitialCSDsData": [[0, 7, 8, 9, 10], [0, 0, 1, 0, 0]]}
        process = {"ShapeFactor": 1, "SolidDensity": 1.4, "SolventDensity": 1,
                   "AntiSolventDensity": 1, "SolvMass": 10, "AntiSolvMass": 0,
                   "SeedMass": 0.001, "PowerRatio": 1}
        solubility = {**dict.fromkeys("abcdefgh", 0), "a": np.log(0.05)}
        n = 50
        _, _, _, state = simulate(parameters, experiment, process, solubility,
                                  {"N": n, "Lmax": 10, "RelTol": 1e-7}, return_history=True)
        edges = np.linspace(0, 10, n + 1)
        resolved = process["SolidDensity"] * (cell_volumes(edges, 1) * np.diff(edges)) @ state[:n] / 1e12
        outflow = state[n + 5]
        self.assertGreater(outflow[-1], process["SeedMass"])
        inventory = resolved + state[n] * state[n + 1] + outflow
        np.testing.assert_allclose(inventory, inventory[0], rtol=0, atol=2e-8)

    def test_growth_reports_nonnegative_csd_despite_roundoff_in_history(self):
        parameters = np.zeros(35)
        parameters[[0, 5, 16, 18, 24, 28]] = -np.inf
        parameters[13] = 1
        experiment = {"TemperatureData": [[0, 1], [20, 20]],
                      "ConcentrationData": [[0], [0.1]],
                      "InitialCSDsData": [[0, 6, 7, 8, 9, 20], [0, 0, 1, 1, 0, 0]]}
        process = {"ShapeFactor": 1, "SolidDensity": 1.4, "SolventDensity": 1,
                   "AntiSolventDensity": 1, "SolvMass": 68, "AntiSolvMass": 0,
                   "SeedMass": 0.001, "PowerRatio": 1}
        solubility = {**dict.fromkeys("abcdefgh", 0), "a": np.log(0.05)}
        n = 10
        _, _, csd, state = simulate(parameters, experiment, process, solubility,
                                    {"N": n, "Lmax": 20}, return_history=True)
        self.assertLess(state[:n, -1].min(), 0)
        np.testing.assert_array_equal(csd[1], np.maximum(state[:n, -1], 0))


if __name__ == "__main__":
    unittest.main()
