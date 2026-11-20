import io
import unittest
from unittest.mock import Mock

import numpy as np

from imed_python.campaign_import import import_run, scale_to_feed


class CampaignImportTests(unittest.TestCase):
    def test_final_size_average_uses_latest_valid_triplets_inside_the_period(self):
        file = Mock()
        samples = (
            "elapsed_s,T,size.value,D50,D90,size.connected\n"
            "0,65,1,2,3,true\n"
            "10,50,10,20,30,true\n"
            "25,40,16,30,45,true\n"
            "20,45,14,26,41,true\n"
            "30,39,30,20,46,true\n"
            "35,38,,35,48,true\n"
            "40,37,22,36,50,false\n"
            "45,36,60,65,75,true\n"
        )
        file.open.side_effect = lambda **kwargs: io.StringIO(samples)
        mapping = {"Temperature": "T", "D10": "size.value", "D50": "D50", "D90": "D90"}
        options = dict(start=10, end=40, concentration_multiplier=1, scale_direction=-1,
                       smoothing=1, initial_concentration=.1, seed_percentiles=None, no_feed=True)

        for requested, count, expected in ((1, 1, [16, 30, 45]), (2, 2, [15, 28, 43]),
                                           (10, 3, np.mean([[10, 20, 30], [14, 26, 41], [16, 30, 45]], axis=0))):
            with self.subTest(requested=requested):
                experiment = import_run(file, mapping, **options, final_size_samples=requested)
                self.assertEqual(experiment["FinalSizeSamplesUsed"], count)
                np.testing.assert_allclose(experiment["FinalD10D50D90"], expected)
                np.testing.assert_array_equal(experiment["D50Data"][1], [20, 26, 30, 20, 35, 36])
        with self.assertRaisesRegex(ValueError, "positive whole number"):
            import_run(file, mapping, **options, final_size_samples=0)

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

    def test_bo_can_import_size_measurements_without_concentration(self):
        file = Mock()
        samples = "elapsed_s,T1.temp_external,D101.value,D501.value,D901.value\n0,50,11,20,30\n10,45,12,25,35\n"
        file.open.side_effect = lambda **kwargs: io.StringIO(samples)
        mapping = {"Temperature": "T1.temp_external", "D10": "D101.value",
                   "D50": "D501.value", "D90": "D901.value"}
        options = {"start": 0, "end": 10, "concentration_multiplier": 1, "scale_direction": -1,
                   "smoothing": 1, "initial_concentration": None, "seed_percentiles": None, "no_feed": True}
        experiment = import_run(file, mapping, **options, require_initial_concentration=False)
        self.assertNotIn("InitialConcentration", experiment)
        np.testing.assert_array_equal(experiment["FinalD10D50D90"], [12, 25, 35])
        with self.assertRaisesRegex(ValueError, "initial concentration"):
            import_run(file, mapping, **options)
