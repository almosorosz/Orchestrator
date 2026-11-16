import unittest

import numpy as np

from imed_python.solver import aggregation_grid, aggregation_update, check_liquid_inventory, profile
from imed_python.validation.cases import Case
from imed_python.validation.qmom import quadrature, solve_reference
from imed_python.validation.run import compare


class NumericalValidationTests(unittest.TestCase):
    def test_quadrature_reconstructs_six_moments(self):
        nodes = np.array([0.15, 0.4, 0.9])
        weights = np.array([0.3, 1.4, 0.6])
        moments = nodes**np.arange(6)[:, None] @ weights
        actual_nodes, actual_weights = quadrature(moments)
        np.testing.assert_allclose(actual_nodes, nodes, atol=1e-12)
        np.testing.assert_allclose(actual_weights, weights, atol=1e-12)

    def test_equal_bin_collision_consumes_two_parents_and_conserves_volume(self):
        volume = np.array([1., 3., 8., 20.])
        pairs = aggregation_grid(volume)
        delta, events = aggregation_update(np.array([0., 2., 0., 0.]), 1., pairs)
        self.assertAlmostEqual(events, 2.)
        self.assertAlmostEqual(delta.sum(), -2.)
        self.assertAlmostEqual(delta[1], -4. + 2. * (8. - 6.) / (8. - 3.))
        self.assertAlmostEqual(volume @ delta, 0.)

    def test_signed_feed_can_empty_vessel_between_knots(self):
        flow = [[0, 10], [-4, 4]]
        with self.assertRaisesRegex(ValueError, "drains"):
            check_liquid_inventory(flow, profile(flow), 10, 5)
        check_liquid_inventory(flow, profile(flow), 10, 15)

    def test_nine_benchmarks(self):
        for operating_profile in ("cooling", "antisolvent", "combined"):
            for mechanism in ("growth", "nucleation", "full"):
                with self.subTest(profile=operating_profile, mechanism=mechanism):
                    case = Case(operating_profile, mechanism)
                    reference = solve_reference(case, np.linspace(0, case.duration, 100))
                    row, _ = compare(case, 80, reference)
                    self.assertTrue(row["passed"], row)
