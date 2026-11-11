"""Tests for the configurable IMED QMOM reference."""
import unittest

import numpy as np

from ..distribution import cell_volumes
from ..solver import simulate
from .cases import Case
from .qmom import solve_ccas_qmom, solve_reference


class ConfigurableQMOMTests(unittest.TestCase):
    def test_zero_order_growth_and_dissolution_approach_equilibrium(self):
        for growth, initial_concentration in ((True, 0.1), (False, 0.0788)):
            with self.subTest(growth=growth):
                case = Case("cooling", "growth", initial_concentration=initial_concentration,
                            solubility_b=0., solubility_d=0.)
                parameters, experiment, process, solubility, settings = case.inputs(160)
                parameters[12:14] = (0., 0.) if growth else (-np.inf, 0.)
                parameters[18:20] = (-np.inf, 0.) if growth else (0., 0.)
                inputs = parameters, experiment, process, solubility, settings
                reference = solve_ccas_qmom(*inputs)
                _, concentration, _, state = simulate(*inputs, return_history=True)
                cells = settings["N"]
                edges = np.linspace(settings["Lmin"], settings["Lmax"], cells + 1)
                solid = (cell_volumes(edges, case.shape) * np.diff(edges)) @ state[:cells]
                api = solid * case.density / 1e12 + state[cells] * state[cells + 1] + state[cells + 5]
                reference_api = reference.crystal_mass + reference.dissolved_mass
                expected_api = (process["SeedMass"] + case.solvent_mass * initial_concentration
                                / (1 - initial_concentration))
                np.testing.assert_allclose(api, expected_api, rtol=0., atol=1e-12)
                np.testing.assert_allclose(reference_api, expected_api, rtol=0., atol=1e-12)
                np.testing.assert_allclose([concentration[1, -1], reference.concentration[-1]],
                                           0.079, rtol=0., atol=1e-9)

    def test_matches_original_combined_profile_reference(self):
        for mechanisms in ("growth", "nucleation", "full"):
            with self.subTest(mechanisms=mechanisms):
                case = Case("combined", mechanisms)
                time = np.linspace(0, case.duration, 100)
                original = solve_reference(case, time)
                # The configurable solver starts from mesh data; refine that mesh
                # before comparing it with the original continuum seed.
                configurable = solve_ccas_qmom(*case.inputs(640), time=time)
                expected = (original.moments * case.number_scale
                            * case.length_scale**np.arange(6)[:, None])
                peak = np.max(np.abs(expected), axis=1)[:, None]
                np.testing.assert_allclose(configurable.moments / peak, expected / peak,
                                           rtol=0, atol=0.003)
                np.testing.assert_allclose(configurable.concentration, original.concentration,
                                           rtol=0, atol=1e-6)
                api = configurable.crystal_mass + configurable.dissolved_mass
                np.testing.assert_allclose(api, api[0], rtol=0, atol=1e-7)
                if mechanisms == "full":
                    self.assertGreater(configurable.aggregation_events[-1], 0)
                    self.assertGreater(configurable.breakup_events[-1], 0)

    def test_seed_with_nonzero_near_zero_volume_has_same_count_and_mass(self):
        parameters, experiment, process, solubility, settings = Case("cooling", "growth").inputs(65)
        parameters[12] = -np.inf
        experiment["InitialCSDsData"] = np.array([
            [0., .5, 1., 2., 5., 10., 65.],
            [0., .01, .04, .08, .05, 0., 0.],
        ])
        reference = solve_ccas_qmom(parameters, experiment, process, solubility, settings)
        _, _, _, history = simulate(parameters, experiment, process, solubility, settings,
                                    return_history=True)
        edges = np.linspace(0., settings["Lmax"], settings["N"] + 1)
        counts = history[:settings["N"], 0] * np.diff(edges)
        np.testing.assert_allclose(reference.number, counts.sum(), rtol=1e-13)
        np.testing.assert_allclose(reference.crystal_mass, process["SeedMass"], rtol=1e-13)

    def test_unseeded_inert_solution_remains_particle_free(self):
        case = Case("combined", "growth")
        parameters, experiment, process, solubility, settings = case.inputs(80)
        process["SeedMass"] = 0
        result = solve_ccas_qmom(parameters, experiment, process, solubility, settings)
        np.testing.assert_array_equal(result.number, 0)
        np.testing.assert_array_equal(result.volume, 0)
        self.assertTrue(np.isnan(result.mean_size).all())
