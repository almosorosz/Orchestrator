import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from experiment.doe_campaign import DoeCampaignService, ProfileAssignment
from experiment.experiment_service import ExperimentService
from experiment.material_balance import pump_mass
from experiment.models import DeviceProfile, ProfilePoint
from experiment.pump_calibration_files import convert_flow, delivered_flow, fit_liquid_flow, seed_dose_command, seed_mass_from_profile
from experiment.pump_profile_data import calibration_equations, dose_duration, read_controller_target
from experiment.recipe import parameter
from gui.pump_calibration_dialog import PumpCalibrationDialog
from gui.pump_profile_dialog import PumpProfileDialog
from hardware_devices.hardware_devices import IsmatecPump


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APP = QApplication.instance() or QApplication([])


class PumpProfileTests(unittest.TestCase):
    def test_solution_log_opens_loaded_run_review_from_calibration_preview(self):
        path = Path(__file__).resolve().parents[1] / "data" / "pump_calibration" / "expttsd1_SOLU.csv"
        dialog = PumpCalibrationDialog({"PSOLU1": {"name": "Solution pump", "type": "masterflex"}})
        try:
            dialog.file.setText(str(path))
            dialog.header.setValue(2)

            def review(reviewer):
                self.assertEqual(reviewer.log["path"], path)
                self.assertEqual(reviewer.time_unit.currentData(), "min")
                self.assertEqual([run["target"] for run in reviewer.plateaus], [20., 50., 70., 100.])
                self.assertTrue(reviewer.use_button.isEnabled())
                self.assertIn("Unchecked runs at 100 RPM", reviewer.status.text())
                highest = reviewer.runs.item(3, 0)
                self.assertEqual(highest.checkState(), Qt.CheckState.Unchecked)
                highest.setCheckState(Qt.CheckState.Checked)
                self.assertFalse(reviewer.use_button.isEnabled())
                highest.setCheckState(Qt.CheckState.Unchecked)
                return QDialog.DialogCode.Accepted

            with patch.object(PumpProfileDialog, "exec", review):
                dialog.preview()
            self.assertTrue(dialog.apply_button.isEnabled(), dialog.summary.text())
            self.assertNotIn("pump_calibration", dialog.configs["PSOLU1"])
            dialog.apply_button.click()
            calibration = dialog.configs["PSOLU1"]["pump_calibration"]
            np.testing.assert_allclose(np.asarray(calibration["points"])[:, 1], [0., 20., 50., 70.])
            self.assertFalse(calibration["dose_curves"][-1]["included"])
        finally:
            dialog.close()

    def test_header_units_and_out_of_order_runs_are_available_for_review(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / "as.csv"
            path.write_text(",,Mass\nTime,File W: AS,File R: Mass\nmin,ml/min,g\n"
                            "0,1,100\n1,1,99\n2,0,99\n3,4,99\n4,4,97\n5,0,97\n6,10,97\n7,10,96.5\n")
            dialog = PumpProfileDialog("AS", {"type": "ismatec"})
            try:
                dialog.load_csv(path)
                self.assertEqual(dialog.time_unit.currentData(), "min")
                self.assertFalse(dialog.use_button.isEnabled())
                dialog.command_unit.setCurrentIndex(dialog.command_unit.findData("mL/min"))
                self.assertFalse(dialog.use_button.isEnabled(), dialog.status.text())
                calibration = dialog.calibration
                self.assertEqual(calibration["output_unit"], "mL/min")
                self.assertEqual([run["included"] for run in calibration["dose_curves"]], [True, True, False])
                # This fixture has a flattening response. The constrained
                # quadratic reaches zero slope at the upper end of the range.
                end_rate = .5 * (.21875 + .5*2) / (.21875**2 + .5**2)
                self.assertAlmostEqual(dose_duration({"type": "ismatec", "pump_calibration": calibration}, 4., 1., preview=True)[0], 60./end_rate)
                self.assertIn("4 mL/min    q = 2", calibration_equations(calibration))
                dialog.command_unit.setCurrentIndex(dialog.command_unit.findData("RPM"))
                self.assertTrue(dialog.use_button.isEnabled(), dialog.status.text())
                self.assertEqual(dialog.calibration["output_unit"], "RPM")
                path.write_text(",0\n")
                self.assertEqual(read_controller_target(path), 0.)
            finally:
                dialog.close()

    def test_legacy_seed_log_keeps_rpm_and_can_reset_its_time_origin(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / "dry_SEED.csv"
            path.write_text(",SEED_PUMP,Mass\nTime,P01:Value (Setpt),File R: Mass\nmin,rpm,g\n"
                            "7,50,0\n10,50,0.0579\n17,50,0.2001\n26,50,0.4409\n")
            dialog = PumpProfileDialog("Seed", {"type": "masterflex"})
            try:
                dialog.load_csv(path)
                self.assertEqual(dialog.calibration["rpm"], 50.)
                self.assertEqual(dialog.calibration["points"][1], [420., 0.])
                self.assertIn("0 ≤ t ≤ 7 min    m = 0", calibration_equations(dialog.calibration))
                dialog.reset_clock.setChecked(True)
                self.assertEqual(dialog.calibration["points"][1], [180., .0579])
            finally:
                dialog.close()

    def test_seed_extrapolation_round_trips_through_a_preparation_profile(self):
        path = Path(__file__).resolve().parents[1] / "data" / "pump_calibration" / "expttsd1_SEED.csv"
        dialog = PumpCalibrationDialog({"SEED": {"name": "Seed pump", "type": "masterflex"}})
        try:
            dialog.file.setText(str(path))
            dialog.header.setValue(2)
            self.assertEqual(dialog.time_unit.currentData(), "s")
            dialog.preview()
            self.assertTrue(dialog.apply_button.isEnabled(), dialog.summary.text())
            dialog.apply_button.click()
            config = dialog.configs["SEED"]
            self.assertEqual(config["seed_calibration"]["seed_fit"]["delay_s"], 7.)
            self.assertEqual(seed_dose_command(config, None, 0.), (50., 0.))
            rpm, duration = seed_dose_command(config, None, .60)
            self.assertGreater(duration, 26.)
            profile = DeviceProfile("SEED", (ProfilePoint(0., rpm), ProfilePoint(duration, 0.)),
                                    interpolation="step", unit="RPM")
            self.assertEqual(seed_mass_from_profile(profile, config, 7.), 0.)
            self.assertAlmostEqual(seed_mass_from_profile(profile, config, duration), .60, places=12)
        finally:
            dialog.close()

    def test_liquid_quadratic_inverts_and_integrates_linear_rpm_ramps(self):
        pump = {"version": 1, "input_unit": "g/h", "output_unit": "RPM", "source_name": "fixture",
                "points": [[0., 0.], [60., 10.], [160., 20.], [300., 30.]]}
        fit_liquid_flow(pump, "masterflex", density=.8)
        config = {"type": "masterflex", "pump_calibration": pump}
        np.testing.assert_allclose(pump["flow_fit"], [4., .2])
        speed = np.array([-30., -18., 0., 18., 30.])
        np.testing.assert_allclose(convert_flow(pump, "masterflex", delivered_flow(config, speed)), speed)
        profile = DeviceProfile("LIQ", (ProfilePoint(0., 0.), ProfilePoint(60., 30.)), unit="RPM")
        self.assertAlmostEqual(pump_mass(profile, config), 2.)
        self.assertIn("mL/min at 0.8 g/mL", calibration_equations(pump))

    def test_reglo_selects_rpm_mode_and_sends_hundredths_of_rpm(self):
        pump = IsmatecPump("AS", "COM1", stage="1")
        pump._send_cmd = Mock(return_value="*")
        self.assertTrue(pump.set_target(12.3))
        self.assertEqual([args.args[0] for args in pump._send_cmd.call_args_list], ["1L", "1K", "1S001230", "1H"])
        rejected = IsmatecPump("AS", "COM1", stage="1")
        rejected._send_cmd = Mock(return_value="#")
        self.assertFalse(rejected.set_target(4.))
        rejected._send_cmd.assert_called_once_with("1L")

    def test_fixed_setting_dose_finishes_early_in_its_stage_and_preserves_amount(self):
        service = ExperimentService()
        service.exp_par.AS_add_max = 4.
        service.exp_par.timeline = {"version": 1, "lanes": [{"id": "AS", "name": "AS", "role": "AS", "initial": 0.}],
            "stages": [{"id": "addition", "name": "Addition", "purpose": "Experiment", "duration": parameter(60.),
                        "actions": {"AS": {"kind": "Dose", "end": parameter(.6), "dose_setting": {"target": 4., "unit": "RPM"}}}}]}
        config = {"type": "ismatec", "pump_calibration": {"version": 1, "input_unit": "g/h", "output_unit": "RPM",
                  "source_name": "fixture", "points": [[0., 0.], [72., 4.], [144., 7.]]}}
        creator = service.generate_initial_doe(1, "LHS")
        campaign = DoeCampaignService(creator).compile((ProfileAssignment("AS", "P"),), {"P": config})
        profile, = campaign.experiments[0].profiles
        self.assertAlmostEqual(pump_mass(profile, config), .6, places=7)
        np.testing.assert_allclose([point.time_s for point in profile.points if point.target > 0], [0., 30.], atol=1e-6)
        self.assertEqual(profile.points[-1].time_s, 60.)
        self.assertEqual(profile.points[-1].target, 0.)
