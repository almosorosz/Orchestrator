import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from experiment.experiment_service import ExperimentService
from experiment.doe_snapshot import snapshot_doe, restore_doe
from app.presets import write_preset, read_preset
from test_doe_campaign import fixture_configs


class SnapshotTests(unittest.TestCase):
    def test_complete_preset_requires_reassignment_before_doe_or_loop_start(self):
        from test_system_topology import APP
        from MAIN_WINDOW import MainWindow
        from experiment.doe_campaign import DoeCampaignService, ProfileAssignment, DEFAULT_ASSIGNMENTS
        from gui.doe_assignment_dialog import DoeAssignmentDialog

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            experiment_path, model_path = Path(root) / "experiments", Path(root) / "model"
            experiment_path.mkdir()
            model_path.mkdir()
            recording = {"output_folder": root, "enabled": True, "sql_enabled": False}
            with patch("gui.recording_settings_dialog.load_recording_settings", return_value=recording), \
                 patch("gui.recording_settings_dialog.save_recording_settings"), \
                 patch("MAIN_WINDOW.QMessageBox.warning") as warning:
                window = MainWindow()
                try:
                    service = window.experiment_service
                    creator = service.generate_initial_doe(2, "GREEDY")
                    configs = fixture_configs()
                    doe = DoeCampaignService(creator)
                    assignments = tuple(ProfileAssignment(role, DEFAULT_ASSIGNMENTS[role]) for role in sorted(doe.required_roles()))
                    monitors = ("SCL_AS",)
                    path = Path(root) / "ready.json"
                    write_preset(path, {"devices": configs, "profiles": {}, "experiments": []},
                                 service.exp_par, assignments,
                                 {"experiment_path": str(experiment_path), "model_path": str(model_path)},
                                 recording, generated_doe=snapshot_doe(creator), monitor_device_ids=monitors)
                    with patch("MAIN_WINDOW.QFileDialog.getOpenFileName", return_value=(str(path), "")), \
                         patch.object(window.settings_page, "render_doe_plots"), \
                         patch.object(service, "generate_initial_doe", side_effect=AssertionError("Must not regenerate")):
                        window.load_setup_preset()
                    warning.assert_not_called()
                    self.assertIsNone(window._assigned_doe_campaign)
                    self.assertEqual(window.CONTROLLER.omni_folder, str(experiment_path))
                    self.assertEqual(window.CONTROLLER.model_folder, str(model_path))
                    self.assertFalse(window.settings_page.start_independent_doe_button.isEnabled())
                    self.assertFalse(any(device.is_connected for device in window.control_panel.hardware.values()))
                    with patch.object(window.control_panel, "start_campaign") as independent, \
                         patch.object(window, "_start_loop_campaign") as loop:
                        window.start_independent_doe()
                        window.start_command()
                        independent.assert_not_called()
                        loop.assert_not_called()

                    def apply_current_devices(dialog):
                        self.assertEqual(set(dialog.assignments()), set(assignments))
                        dialog.preview()
                        dialog.accept()
                        return dialog.result()

                    with patch.object(DoeAssignmentDialog, "exec", apply_current_devices):
                        window.assign_doe_devices()
                    campaign = window._assigned_doe_campaign
                    self.assertEqual(len(campaign.experiments), 2)
                    self.assertEqual(campaign.monitor_device_ids, ())
                    self.assertTrue(window.settings_page.start_independent_doe_button.isEnabled())
                    with patch.object(window.control_panel, "start_campaign") as start:
                        window.start_independent_doe()
                        start.assert_called_once_with(campaign)
                    with patch.object(window, "_start_loop_campaign") as start:
                        window.stop_command()
                        window.start_command()
                        start.assert_called_once_with(campaign)
                finally:
                    window.close()
                    APP.processEvents()

    def test_main_window_preset_restores_generated_conditions(self):
        from test_system_topology import APP
        from MAIN_WINDOW import MainWindow
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            settings = {"output_folder": root, "enabled": True, "sql_enabled": False, "remote_sql_url": ""}
            with patch("gui.recording_settings_dialog.load_recording_settings", return_value=settings), \
                 patch("gui.recording_settings_dialog.save_recording_settings"), \
                 patch("MAIN_WINDOW.QMessageBox.warning") as warning:
                window = MainWindow()
                try:
                    window.create_experiments = window.experiment_service.generate_initial_doe(3, "RAND")
                    original = snapshot_doe(window.create_experiments)
                    filename = str(Path(root) / "setup.json")
                    with patch("MAIN_WINDOW.QFileDialog.getSaveFileName", return_value=(filename, "")):
                        window.save_setup_preset()
                    window.create_experiments = None
                    with patch("MAIN_WINDOW.QFileDialog.getOpenFileName", return_value=(filename, "")), \
                         patch.object(window.settings_page, "render_doe_plots"), \
                         patch.object(window.experiment_service, "generate_initial_doe", side_effect=AssertionError("Must not resample")):
                        window.load_setup_preset()
                    warning.assert_not_called()
                    self.assertEqual(snapshot_doe(window.create_experiments), original)
                    self.assertIsNone(window._assigned_doe_campaign)
                finally:
                    window.close()
                    APP.processEvents()

    def test_random_conditions_and_all_profile_tables_are_restored_without_resampling(self):
        service = ExperimentService()
        creator = service.generate_initial_doe(3, "RAND")
        snapshot = snapshot_doe(creator)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            filename = Path(root) / "preset.json"
            write_preset(filename, {"devices": fixture_configs(), "profiles": {}, "experiments": []}, service.exp_par, (), {},
                         {"output_folder": root, "sql_enabled": False}, generated_doe=snapshot)
            data, params, _ = read_preset(filename)
            restored_service = ExperimentService()
            restored_service.exp_par = params
            restored_service._build_doe_dependencies()
            with patch.object(restored_service, "generate_initial_doe", side_effect=AssertionError("Must not re-sample")):
                restored = restore_doe(restored_service, data["generated_doe"])
            np.testing.assert_array_equal(restored.DoE_samples, creator.DoE_samples)
            self.assertEqual(restored.parameter_snapshots, creator.parameter_snapshots)
            for table in snapshot["storage"]:
                for name, original in getattr(creator.exp_storage, table).items():
                    pd.testing.assert_frame_equal(getattr(restored.exp_storage, table)[name], original, check_dtype=False)
            self.assertEqual(restored.actually_sampled_experiments, 3)

    def test_old_preset_without_generated_doe_still_loads(self):
        service = ExperimentService()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            filename = Path(root) / "old.json"
            write_preset(filename, {"devices": fixture_configs(), "profiles": {}, "experiments": []}, service.exp_par, (), {}, {"output_folder": root})
            data = json.loads(filename.read_text())
            data["version"] = 1
            data.pop("generated_doe")
            filename.write_text(json.dumps(data))
            loaded, _, _ = read_preset(filename)
            self.assertIsNone(restore_doe(service, loaded.get("generated_doe")))
