"""External templates load through the normal runner; restored monitors remain visible."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from time import monotonic, sleep

import pandas as pd
from control_panel.control_panel import HARDWARE_TYPES, create_hardware
from experiment_dashboard.experiment_dashboard import ExperimentWorker
from gui.doe_assignment_dialog import DoeAssignmentDialog
from gui.external_model_info import experiment_template
from gui.plot_renderer import OriginalPlotRenderer
from test_recipe_editor import APP, CONFIGS
import test_recipe_editor as recipe_fixture


MONITORS = ("FBRM1", "D101", "D501", "D901")


class ExternalModelInfoTests(unittest.TestCase):
    def wait_for(self, condition):
        deadline = monotonic() + 20
        while monotonic() < deadline:
            APP.processEvents()
            if condition():
                return True
            sleep(.01)
        return False

    def devices(self):
        configs = deepcopy(CONFIGS)
        for key in MONITORS:
            configs[key] = {"name": key, "type": "opcua", "simulated": True, "connection": {},
                            "plot_index": 3 if key == "FBRM1" else 4, "connectivity": {"role": "read_only"}}
        hardware = create_hardware(configs, HARDWARE_TYPES)
        for device in hardware.values():
            device.connect()
        return configs, hardware

    def test_saved_empty_monitor_filter_matches_checkbox_and_rechecking_includes_live_channels(self):
        configs, hardware = self.devices()
        creator = recipe_fixture.RecipeEditorTests().service().generate_initial_doe(1, "LHS")
        dialog = DoeAssignmentDialog(creator, configs, hardware)
        try:
            dialog.restore_monitor_selection([])
            self.assertFalse(dialog.include_monitors.isChecked())
            dialog.include_monitors.setChecked(True)
            dialog.preview()
            self.assertIsNotNone(dialog.campaign, dialog.summary.text())
            self.assertEqual(set(dialog.campaign.monitor_device_ids), set(MONITORS))
            self.assertTrue(all(key in dialog.monitor_summary.text() for key in MONITORS))
            renderer = OriginalPlotRenderer(device_configs=configs)
            self.assertTrue(all((key, "value") in renderer.lines for key in MONITORS))
            worker = ExperimentWorker(hardware, configs, list(dialog.campaign.experiments), speed=100.,
                                      end_hold_s=0., campaign=dialog.campaign)
            packets = []
            worker.status_updated.connect(packets.append)
            worker.start_experiment()
            self.assertEqual(worker.outcome.value, "completed", worker.error_message)
            self.assertTrue(packets)
        finally:
            dialog.close()
            for device in hardware.values():
                device.disconnect()

    def test_restored_subset_and_disconnected_probes_are_explicit(self):
        configs, hardware = self.devices()
        creator = recipe_fixture.RecipeEditorTests().service().generate_initial_doe(1, "LHS")
        dialog = DoeAssignmentDialog(creator, configs, hardware)
        try:
            dialog.restore_monitor_selection(["FBRM1"])
            dialog.preview()
            self.assertEqual(dialog.campaign.monitor_device_ids, ("FBRM1",))
            dialog.include_monitors.setChecked(False)
            dialog.include_monitors.setChecked(True)
            hardware["D901"].disconnect()
            dialog.preview()
            self.assertEqual(set(dialog.campaign.monitor_device_ids), set(MONITORS) - {"D901"})
            self.assertIn("Disconnected (excluded): D901", dialog.monitor_summary.text())
        finally:
            dialog.close()
            for device in hardware.values():
                device.disconnect()

    def test_info_template_exports_and_loads_actual_experiments(self):
        from MAIN_WINDOW import MainWindow
        window = MainWindow()
        try:
            with TemporaryDirectory(dir=Path.cwd()) as folder:
                root = Path(folder)
                window.settings_page.exp_path_input.setText(str(root))
                window.settings_page.MODEL_path_input.setText(str(root))
                configs = {"TEMP": deepcopy(CONFIGS["TEMP"])}
                frame = pd.DataFrame({"time_s": [0., 30.], "TEMP": [25., 24.]})
                panel = window.control_panel
                panel.apply_loaded_configuration(configs, {}, [frame.to_dict(orient="split")])
                info = window.settings_page.show_external_model_info()
                self.assertEqual(info.path_fields["experiment_root"].text(), str(root))
                self.assertIn("next_experiments.json", info.pages["Files and folders"].toPlainText())
                self.assertNotIn("Automatic connector", info.pages)
                info.copy_prompt()
                self.assertIn("next_experiments.json is not automatically imported", APP.clipboard().text())
                self.assertIn("Do not request, disclose or modify CrystalLoop's codebase", APP.clipboard().text())
                self.assertNotIn("proposed extension", " ".join(page.toPlainText() for page in info.pages.values()))
                filename = root / "next_experiments.json"
                with patch("gui.external_model_info.QFileDialog.getSaveFileName", return_value=(str(filename), "")), \
                     patch("gui.external_model_info.QMessageBox.information"):
                    self.assertEqual(info.export_template(), filename)
                payload = json.loads(filename.read_text())
                self.assertEqual(payload["experiments"][0]["columns"], ["time_s", "TEMP"])
                with patch("control_panel.control_panel.QFileDialog.getOpenFileName", return_value=(str(filename), "")), \
                     patch("control_panel.control_panel.QMessageBox.information"):
                    panel.load_configuration()
                pd.testing.assert_frame_equal(panel.experiments[0], frame)
                info.close()
        finally:
            window.close()
            APP.processEvents()

    def test_template_requires_real_defined_profiles_and_rejects_read_only_target_columns(self):
        with self.assertRaisesRegex(ValueError, "First generate"):
            experiment_template({"devices": CONFIGS, "experiments": []})
        frame = pd.DataFrame({"time_s": [0., 1.], "probe": [1., 2.]})
        with self.assertRaisesRegex(ValueError, "Read-only"):
            experiment_template({"devices": {"probe": {"type": "opcua"}}, "experiments": [frame.to_dict(orient="split")]})

    def test_control_panel_start_uses_general_exp_folder_and_leaves_model_unused(self):
        from MAIN_WINDOW import MainWindow
        window = MainWindow()
        try:
            with TemporaryDirectory(dir=Path.cwd()) as directory:
                root = Path(directory)
                experiments = root / "general_experiments"
                experiments.mkdir()
                unused_model = root / "unused_models"
                window.settings_page.exp_path_input.setText(str(experiments))
                window.settings_page.MODEL_path_input.setText(str(unused_model))
                panel = window.control_panel
                configs = {"TEMP": deepcopy(CONFIGS["TEMP"])}
                frame = pd.DataFrame({"time_s": [0., 1.], "TEMP": [25., 25.]})
                panel.apply_loaded_configuration(configs, {}, [frame.to_dict(orient="split")], str(root / "old_CP_path"))
                self.assertEqual(panel.control_output_path.text(), str(experiments))
                panel.hardware["TEMP"].connect()
                panel.simulation_speed.setValue(100.)
                panel.start_experiment()
                dashboard = window.experiment_page.dashboard
                self.assertIsNotNone(dashboard)
                self.assertEqual(dashboard.recording_settings["output_folder"], str(experiments))
                self.assertTrue(self.wait_for(lambda: hasattr(dashboard, "last_result") and dashboard.thread is None))
                self.assertEqual(dashboard.last_result["status"], "completed")
                self.assertTrue(list(experiments.glob("campaign_*/*/execution.csv")))
                self.assertFalse(unused_model.exists())
                window.end_experiment_session()
        finally:
            window.close()
            APP.processEvents()

    def test_external_dashboard_refreshes_stale_controller_paths_before_execution(self):
        from MAIN_WINDOW import MainWindow
        from app.controller import Stage
        window = MainWindow()
        try:
            with TemporaryDirectory(dir=Path.cwd()) as directory:
                root = Path(directory)
                old_exp, old_model = root / "old_exp", root / "old_model"
                exp, model = root / "exp", root / "model"
                for path in (old_exp, old_model, exp, model):
                    path.mkdir()
                settings = window.settings_page
                settings.exp_path_input.setText(str(old_exp))
                settings.MODEL_path_input.setText(str(old_model))
                settings.exp_number_input.setText("1")
                self.assertTrue(window.define_command())
                settings.exp_path_input.setText(str(exp))
                settings.MODEL_path_input.setText(str(model))
                settings.loop_backend.setCurrentIndex(settings.loop_backend.findData("external"))
                panel = window.control_panel
                frame = pd.DataFrame({"time_s": [0., 1.], "TEMP": [25., 25.]})
                panel.apply_loaded_configuration({"TEMP": deepcopy(CONFIGS["TEMP"])}, {}, [frame.to_dict(orient="split")])
                panel.hardware["TEMP"].connect()
                panel.simulation_speed.setValue(100.)
                window.start_command()
                self.assertEqual(window.CONTROLLER.omni_folder, str(exp))
                self.assertEqual(window.CONTROLLER.model_folder, str(model))
                self.assertTrue(self.wait_for(lambda: bool(list(model.glob("*/campaign_input.json")))))
                self.assertEqual(window.execution.monitor_mode, Stage.MODEL)
                self.assertTrue(list(exp.glob("campaign_*/*/execution.csv")))
                self.assertFalse(list(old_exp.iterdir()))
                self.assertFalse(list(old_model.iterdir()))
                window.stop_command()
                self.assertTrue(self.wait_for(lambda: not panel.hardware_manager.is_busy))
                window.end_experiment_session()
        finally:
            window.stop_command()
            window.close()
            APP.processEvents()
