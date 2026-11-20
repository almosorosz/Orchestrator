import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

from experiment.pump_calibration_files import load_calibration, convert_flow, validate_calibration
from experiment.doe_campaign import DoeCampaignService
from test_doe_campaign import fixture_creator, fixture_configs, fixture_assignments

APP = QApplication.instance() or QApplication([])


class CalibrationTests(unittest.TestCase):
    def test_table_snapshot_interpolates_signed_values_and_rejects_extrapolation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            file = Path(root) / "pump.csv"
            file.write_text("flow_g_h,rpm\n0,0\n60,10\n120,20\n")
            calibration = load_calibration(file, "masterflex", "flow_g_h", "rpm")
            file.unlink()  # Configuration/preset retains the measured points.
            np.testing.assert_allclose(convert_flow(calibration, "masterflex", [-90, 0, 90]), [-15, 0, 15])
            with self.assertRaisesRegex(ValueError, "range"):
                convert_flow(calibration, "masterflex", [121])
            with self.assertRaisesRegex(ValueError, "convert"):
                validate_calibration(calibration, "ismatec")

    def test_mass_log_uses_time_units_and_measured_mass_slopes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            file = Path(root) / "log.csv"
            file.write_text("metadata,,\nTime,rpm,mass_g\nmin,rpm,g\n0,10,100\n1,10,99\n2,10,98\n3,20,97\n4,20,95\n5,20,93\n")
            calibration = load_calibration(file, "masterflex", "", "rpm", mode="mass_log", header_row=1,
                time_column="Time", time_unit="min", mass_column="mass_g")
            np.testing.assert_allclose(calibration["points"], [[0, 0], [60, 10], [120, 20]], atol=1e-8)

    def test_plot_shows_each_measured_plateau_against_the_conversion_curve(self):
        from gui.pump_calibration_dialog import calibration_figure

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            file = Path(root) / "repeats.csv"
            file.write_text("time_min,rpm,mass_g\n0,10,100\n1,10,99\n2,20,98\n3,20,95\n4,10,93\n5,10,91\n")
            calibration = load_calibration(file, "masterflex", "", "rpm", mode="mass_log",
                                           time_column="time_min", mass_column="mass_g")
            axis, = calibration_figure(calibration).axes
            np.testing.assert_allclose(axis.lines[0].get_xydata(), [[0, 0], [10, 90], [20, 180]])
            np.testing.assert_allclose(axis.collections[0].get_offsets(), [[10, 60], [20, 180], [10, 120]])
            self.assertEqual(axis.collections[1].get_label(), "Added zero point")

    def test_dialog_plots_a_csv_preview_and_saved_dry_seed_curve(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from PySide6.QtWidgets import QDialog, QTabWidget
        from experiment.pump_calibration_files import load_seed_calibration
        from gui.pump_calibration_dialog import PumpCalibrationDialog

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            root = Path(root)
            liquid = root / "liquid.csv"
            liquid.write_text("flow_g_h,rpm\n60,10\n120,20\n")
            seed = root / "seed.csv"
            seed.write_text("time_min,dry_seed_g\n1,0.15\n2,0.4\n")
            configs = fixture_configs()
            config = configs["PMP_SEED"]
            config["seed_calibration"] = load_seed_calibration(seed, config, "time_min", "dry_seed_g", 50., time_unit="min")
            seed.unlink()
            dialog = PumpCalibrationDialog(configs)
            try:
                dialog.device.setCurrentIndex(dialog.device.findData("PMP_SOLU"))
                self.assertFalse(dialog.plot_button.isEnabled())
                dialog.device.setCurrentIndex(dialog.device.findData("PMP_SEED"))
                self.assertTrue(dialog.plot_button.isEnabled())
                dialog.file.setText(str(liquid))
                dialog.read_columns()
                dialog.preview()
                dialog.plot_button.click()
                popup = dialog.findChild(QDialog, "pumpCalibrationPlot")
                tabs = popup.findChild(QTabWidget)
                self.assertEqual(tabs.tabText(tabs.currentIndex()), "Liquid flow")
                axes = [canvas.figure.axes[0] for canvas in tabs.findChildren(FigureCanvasQTAgg)]
                self.assertEqual({axis.get_xlabel() for axis in axes}, {"Pump command (RPM)", "Pumping time (min)"})
                self.assertTrue(any(axis.get_ylabel() == "Dry seed mass at 50 RPM (g)" for axis in axes))
                self.assertNotIn("pump_calibration", dialog.configs["PMP_SEED"])
                popup.close()
            finally:
                dialog.close()

    def test_ismatec_rpm_calibration_drives_generated_device_targets(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            file = Path(root) / "ismatec.csv"
            file.write_text("flow_g_h,rpm\n0,0\n60,2\n120,4\n")
            calibration = load_calibration(file, "ismatec", "flow_g_h", "rpm")
            configs = fixture_configs()
            configs["PMP_AS"]["pump_calibration"] = calibration
            service = DoeCampaignService(fixture_creator())
            campaign = service.compile(fixture_assignments(), configs)
            profile = next(p for p in campaign.experiments[0].profiles if p.device_id == "PMP_AS")
            self.assertAlmostEqual(profile.points[0].target, 2.)
            self.assertEqual(profile.unit, "RPM")
            details = next(d for d in service.assignment_details(fixture_assignments(), configs) if d["device_id"] == "PMP_AS")
            self.assertEqual(details["calibration_sha256"], calibration["sha256"])
            self.assertEqual(details["calibration_file"], "ismatec.csv")

    def test_bad_units_and_nonmonotonic_calibration_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            file = Path(root) / "bad.csv"
            file.write_text("flow_g_h,rpm\n0,0\n60,10\n120,5\n")
            with self.assertRaisesRegex(ValueError, "increase"):
                load_calibration(file, "masterflex", "flow_g_h", "rpm")
            file.write_text("flow_g_h,ml_min\n0,0\n60,10\n120,20\n")
            with self.assertRaisesRegex(ValueError, "mL/min"):
                load_calibration(file, "ismatec", "flow_g_h", "ml_min")

    def test_dialog_applies_snapshot_and_control_panel_config_preserves_it(self):
        from MAIN_WINDOW import MainWindow
        from gui.pump_calibration_dialog import PumpCalibrationDialog
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            file = Path(root) / "pump.csv"
            file.write_text("flow_g_h,rpm\n0,0\n60,10\n120,20\n")
            configs = fixture_configs()
            dialog = PumpCalibrationDialog(configs)
            try:
                dialog.device.setCurrentIndex(dialog.device.findData("PMP_SOLU"))
                dialog.file.setText(str(file))
                dialog.read_columns()
                dialog.preview()
                self.assertIsNotNone(dialog.candidate, dialog.summary.text())
                dialog.apply()
                self.assertIn("pump_calibration", dialog.configs["PMP_SOLU"])
                settings = {"enabled": True, "sql_enabled": False, "output_folder": root, "remote_sql_url": ""}
                with patch("gui.recording_settings_dialog.load_recording_settings", return_value=settings):
                    window = MainWindow()
                    try:
                        window.control_panel.apply_loaded_configuration(dialog.configs, {}, [])
                        restored, _ = window.control_panel.collect_current_configuration()
                        self.assertEqual(restored["PMP_SOLU"]["pump_calibration"]["points"], [[0., 0.], [60., 10.], [120., 20.]])
                        self.assertFalse(window.control_panel.hardware["PMP_SOLU"].is_connected)
                    finally:
                        window.close()
                        APP.processEvents()
            finally:
                dialog.close()
