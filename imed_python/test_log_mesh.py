import unittest
from importlib.util import find_spec

import numpy as np

from imed_python.comparison import compare_model, initial_api_mass
from imed_python.distribution import cell_volumes, d_values, volume_distribution
from imed_python.estimation import FitObjective
from imed_python.mesh_grid import grid_edges
from imed_python.mesh_validation import validate_meshes
from imed_python.pe_settings import (
    PE_PBM_KEY, PYFIT_DEFAULTS, WEIGHT_DEFAULTS, effective_pe_pbm, pe_file, read_pe_file,
)
from imed_python.solver import simulate
from imed_python.validation.cases import Case


class LogMeshTests(unittest.TestCase):
    def test_shifted_spacing_keeps_zero_and_concentrates_bins_at_the_lower_end(self):
        settings = {"Lmin": 0., "Lmax": 250., "N": 250, "GridSpacing": "log", "LogGridOffset": 10.}
        edges = grid_edges(settings)
        self.assertEqual(len(edges), 251)
        np.testing.assert_equal(edges[[0, -1]], [0., 250.])
        widths = np.diff(edges)
        self.assertTrue(np.all(np.diff(widths) > 0))
        self.assertLess(widths[0], .2)
        np.testing.assert_allclose(np.diff(np.log(edges + 10)), np.log(26) / 250)
        np.testing.assert_equal(grid_edges({}), np.linspace(0, 250, 251))

    def test_volume_percentiles_include_the_full_smallest_cell(self):
        edges = np.array([0., .01, .1])
        sizes = d_values([.005, .055], [1., 0.], edges=edges)
        np.testing.assert_allclose(sizes, [.001, .005, .009])

    def test_pe_settings_round_trip_and_archived_solver_keeps_uniform_mesh(self):
        settings = {"GridSpacing": "log", "LogGridOffset": 5.}
        saved, _, _ = read_pe_file(pe_file(settings, PYFIT_DEFAULTS, WEIGHT_DEFAULTS))
        self.assertEqual(saved, settings)
        self.assertEqual(effective_pe_pbm({PE_PBM_KEY: saved})["GridSpacing"], "log")
        archived = {PE_PBM_KEY: saved, "pbmsetting": {"Numerics": "matlab_pe"}}
        np.testing.assert_equal(grid_edges(effective_pe_pbm(archived)), np.linspace(0, 250, 251))
        with self.assertRaisesRegex(ValueError, "positive log offset"):
            read_pe_file(pe_file({**settings, "LogGridOffset": 0.}, PYFIT_DEFAULTS, WEIGHT_DEFAULTS))

    def test_scipy_fit_and_validation_use_the_same_log_edges(self):
        parameters, experiment, process, solubility, settings = Case("cooling", "growth").inputs(40)
        settings.update(GridSpacing="log", LogGridOffset=5.)
        edges = grid_edges(settings)
        _, _, csd = simulate(parameters, experiment, process, solubility, settings)
        np.testing.assert_array_equal(csd[0], edges[1:] - np.diff(edges) / 2)
        measured_size = np.linspace(edges[0], edges[-1], 23)
        measured_density = np.interp(measured_size, csd[0], csd[1] * cell_volumes(edges, process["ShapeFactor"]))
        experiment["FinalCSDsData"] = np.vstack((measured_size, measured_density))
        objective = FitObjective([experiment], [process], solubility, settings, [0, 0, 1], "CSD")
        self.assertEqual(objective.residual_count, len(measured_size))
        np.testing.assert_allclose(objective.residuals(parameters), 0., atol=1e-12)

        model = {"paraset": np.column_stack((np.zeros(len(parameters)), parameters, parameters, parameters))}
        case = {"ModelParameter": [model], "ExpData": [experiment], "ProcInf": [process],
                "Solubility": solubility, "pbmsetting": settings}
        result = validate_meshes(case, 0, 0, [20, 40])
        refined = result["runs"][1]
        distribution, sizes = volume_distribution(csd, process, settings)
        np.testing.assert_allclose(refined["density"], distribution[1])
        np.testing.assert_allclose(refined["percentiles"], sizes)
        self.assertAlmostEqual(distribution[1] @ np.diff(edges), 100.)
        self.assertLess(refined["mass_error_ppm"], .01)
        comparison = compare_model(model, experiment, process, solubility, settings)
        np.testing.assert_equal(comparison["bounds"], edges)
        np.testing.assert_allclose(comparison["pbm"]["percentiles"][:, -1], sizes)
        self.assertIn("qmom", comparison, comparison.get("qmom_error"))

    @unittest.skipUnless(find_spec("jax") and find_spec("diffrax"), "JAX and Diffrax are optional")
    def test_jax_and_scipy_agree_and_conserve_mass_on_the_log_grid(self):
        parameters, experiment, process, solubility, settings = Case("combined", "full").inputs(60)
        settings.update(GridSpacing="log", LogGridOffset=5.)
        inputs = parameters, experiment, process, solubility
        scipy = simulate(*inputs, settings, return_history=True)
        jax = simulate(*inputs, {**settings, "Numerics": "jax", "JaxDevice": "cpu"}, return_history=True)
        edges = grid_edges(settings)
        widths = np.diff(edges)
        np.testing.assert_equal(jax[2][0], scipy[2][0])
        np.testing.assert_allclose(jax[0], scipy[0], rtol=1e-3)
        np.testing.assert_allclose(jax[1], scipy[1], rtol=1e-4)
        np.testing.assert_allclose(jax[2][1], scipy[2][1], atol=1e-3 * scipy[2][1].max())
        for _, _, _, state in (scipy, jax):
            cells = settings["N"]
            solid = (cell_volumes(edges, process["ShapeFactor"]) * widths) @ state[:cells]
            total = state[cells] * state[cells + 1] + process["SolidDensity"] * solid / 1e12 + state[cells + 5]
            np.testing.assert_allclose(total, initial_api_mass(experiment, process), atol=1e-10, rtol=0)
