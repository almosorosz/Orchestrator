import io
import unittest
from unittest.mock import Mock

import numpy as np

from imed_python.campaign_import import import_run, scale_to_feed


class CampaignImportTests(unittest.TestCase):
    def test_all_recorded_channels_and_mass_preserving_feed(self):
        file = Mock()
        file.open.return_value = io.StringIO(
            "elapsed_s,T1.temp_external,IR1.value,FBRM1.value,D101.value,D501.value,D901.value,WAS1.weight_g\n"
            "10,50,10,100,11,20,33,100\n"
            "20,45,9,120,12,22,34,99\n"
            "30,40,8,130,13,24,37,97\n"
            "40,35,7,140,,26,39,96\n"
        )
        mapping = dict(zip(("Temperature", "Concentration", "FBRM", "D10", "D50", "D90", "AS scale"),
                           ("T1.temp_external", "IR1.value", "FBRM1.value", "D101.value",
                            "D501.value", "D901.value", "WAS1.weight_g")))
        exp = import_run(file, mapping, start=10, end=40, concentration_multiplier=0.01,
                         scale_direction=-1, smoothing=1, initial_concentration=None,
                         seed_percentiles=(11, 20, 33), no_feed=False)
        np.testing.assert_array_equal(exp["TemperatureData"], [[0, 10, 20, 30], [50, 45, 40, 35]])
        np.testing.assert_allclose(exp["ConcentrationData"][1], [.1, .09, .08, .07])
        np.testing.assert_array_equal(exp["CountsData"][1], [100, 120, 130, 140])
        np.testing.assert_array_equal(exp["FinalD10D50D90"], [13, 24, 37])
        np.testing.assert_array_equal(exp["D50Data"][1], [20, 22, 24, 26])
        self.assertAlmostEqual(np.trapezoid(exp["FlowRateData"][1], exp["FlowRateData"][0]), 4.)
        self.assertEqual(exp["FinalCSDsData"].size, 0)

    def test_scale_direction_and_no_negative_feed_from_reversals(self):
        scale = np.array([[0, 2, 5, 8], [10, 12, 11.5, 14]])
        added, feed = scale_to_feed(scale, direction=1, smoothing=1)
        np.testing.assert_array_equal(added[1], [0, 2, 2, 4])
        self.assertTrue((feed[1] >= 0).all())
        self.assertAlmostEqual(np.trapezoid(feed[1], feed[0]), 4)
        with self.assertRaisesRegex(ValueError, "opposite"):
            scale_to_feed(scale, direction=-1, smoothing=1)
