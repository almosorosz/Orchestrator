"""Linked and independent profiles survive sampling, presets and execution."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QDialog

from app.presets import read_preset, write_preset
from experiment.doe_campaign import DoeCampaignService, ProfileAssignment
from experiment.doe_snapshot import restore_doe, snapshot_doe
from experiment.experiment_service import ExperimentService
from experiment.recipe import parameter, validate_recipe
from gui.device_profile_dialog import DeviceProfileDialog
from gui.recipe_editor import ExperimentDefinitionDialog, ParameterTable
from test_recipe_editor import APP, CONFIGS, template


def action(value, kind="Hold", start=25., linked=False):
    result = {"name": kind, "kind": kind, "end": parameter(value), "start": parameter(start), "control_mode": "Process"}
    if linked:
        result["end"]["equal_previous"] = True
    if kind == "Ramp":
        result["start"]["equal_previous"] = True
    return result


def section(name, duration, value, **kwargs):
    return {"id": name, "name": name, "duration": parameter(duration), **action(value, **kwargs)}


class ProfileSynchronizationTests(unittest.TestCase):
    def service(self, recipe):
        service = ExperimentService()
        service.exp_par.v_cool_max = service.exp_par.v_heat_max = 60.
        service.exp_par.timeline = recipe
        return service

    def test_alternating_cooling_end_is_inherited_by_equilibration_and_saved(self):
        recipe = template()
        recipe["stages"][1]["actions"]["T"]["end"] = parameter(24., "Alternate", 23., 24.)
        hold = action(19., linked=True)
        # Linked fields must not create an independent DoE dimension even if
        # they previously had sampling enabled.
        hold["end"].update(mode="DoE", min=18., max=20.)
        recipe["stages"][2]["actions"]["T"] = hold
        service = self.service(recipe)
        creator = service.generate_initial_doe(4, "LHS")
        self.assertEqual(creator.dimensions, 0)
        for index, (name, frame) in enumerate(creator.exp_storage.data.items()):
            expected = 23. if index % 2 == 0 else 24.
            times = frame["t"].to_numpy() * 3600
            self.assertEqual(np.interp(8., times, frame["T"]), expected)
            self.assertEqual(creator.parameter_snapshots[name]["eq.T.end"]["value"], expected)
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "linked.json"
            write_preset(path, {"devices": CONFIGS}, service.exp_par, (), {},
                         {"enabled": True, "sql_enabled": False, "output_folder": folder}, generated_doe=snapshot_doe(creator))
            saved, params, _ = read_preset(path)
            self.assertTrue(params.timeline["stages"][2]["actions"]["T"]["end"]["equal_previous"])
            service.exp_par = params
            service._build_doe_dependencies()
            self.assertEqual(snapshot_doe(restore_doe(service, saved["generated_doe"])), snapshot_doe(creator))

    def independent_recipe(self):
        recipe = template()
        recipe["stages"] = []
        recipe["lanes"] = recipe["lanes"][:2] + [{"id": "STR", "role": "stirring_rate", "name": "Stirrer",
                                                     "device_id": "STIR", "initial": 300.}]
        cooling = section("cool", 10., 24., kind="Ramp")
        cooling["duration"] = parameter(10., "DoE", 10., 20.)
        cooling["end"] = parameter(24., "Alternate", 23., 24.)
        recipe["lanes"][0]["profile"] = [cooling, section("equilibrate", 5., 19., linked=True)]
        recipe["lanes"][1]["profile"] = [section("dose", 3., .03, kind="Dose")]
        recipe["lanes"][2]["profile"] = [section("constant", 1., 300.)]
        return recipe

    def test_independent_sampled_durations_extend_constants_and_finish_mass_addition(self):
        recipe = self.independent_recipe()
        service = self.service(recipe)
        creator = service.generate_initial_doe(2, "FACT")
        self.assertEqual(creator.dimensions, 1)
        self.assertEqual([frame["t"].iloc[-1] * 3600 for frame in creator.exp_storage.data.values()], [15., 25.])
        for index, frame in enumerate(creator.exp_storage.data.values()):
            self.assertTrue(np.all(frame["STR"] == 300.))
            self.assertEqual(frame["T"].iloc[-1], 23. if index == 0 else 24.)
            self.assertEqual(frame["AS"].iloc[-1], 0.)
            self.assertAlmostEqual(np.trapezoid(frame["AS"], frame["t"]), .03, places=7)
        configs = {**CONFIGS, "STIR": {"name": "Stirrer", "type": "jkem", "simulated": True}}
        assignments = tuple(ProfileAssignment(lane["id"], lane["device_id"]) for lane in recipe["lanes"])
        campaign = DoeCampaignService(creator).compile(assignments, configs)
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "independent.json"
            write_preset(path, {"devices": configs}, service.exp_par, assignments, {},
                         {"enabled": True, "sql_enabled": False, "output_folder": folder}, generated_doe=snapshot_doe(creator))
            saved, params, roles = read_preset(path)
            service.exp_par = params
            service._build_doe_dependencies()
            restored = restore_doe(service, saved["generated_doe"])
            restored_campaign = DoeCampaignService(restored).compile(roles, configs)
            self.assertEqual(restored_campaign.experiments[1].profiles, campaign.experiments[1].profiles)
        from experiment_dashboard.experiment_dashboard import ExperimentWorker
        from hardware_devices.simulated import SimulatedDevice
        devices = {key: SimulatedDevice(key, config["type"], noise=0.) for key, config in configs.items()}
        for device in devices.values():
            device.connect()
        worker = ExperimentWorker(devices, configs, list(campaign.experiments), speed=100., end_hold_s=0., campaign=campaign)
        worker.start_experiment()
        self.assertEqual(worker.outcome.value, "completed", worker.error_message)

    def test_one_shared_action_can_span_all_stages_and_the_longest_device_profile(self):
        recipe = template()
        recipe["lanes"].append({"id": "STR", "role": "stirring_rate", "name": "Stirrer", "initial": 300., "device_id": ""})
        constant = action(300.)
        constant["until_end"] = True
        recipe["stages"][0]["actions"]["STR"] = constant
        recipe["lanes"][1]["profile"] = [section("off", 20., 0.)]
        recipe["stages"][1]["actions"].pop("AS")
        creator = self.service(recipe).generate_initial_doe(1, "LHS")
        frame = creator.exp_storage.data["exp1"]
        self.assertEqual(frame["t"].iloc[-1] * 3600, 20.)
        self.assertTrue(np.all(frame["STR"] == 300.))
        recipe["stages"][2]["actions"]["STR"] = action(400.)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_recipe(recipe)

    def test_editor_checkbox_and_device_profile_preserve_linked_controls(self):
        recipe = self.independent_recipe()
        service = self.service(recipe)
        dialog = ExperimentDefinitionDialog(service.exp_par, CONFIGS)
        try:
            lane = dialog.editor.recipe["lanes"][0]
            profile = DeviceProfileDialog(lane, [], dialog)
            profile.section_list.setCurrentRow(1)
            self.assertTrue(profile.table.links[1].isChecked())
            self.assertFalse(profile.table.cellWidget(1, 2).isEnabled())
            profile.accept()
            lane["profile"] = profile.sections
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted, dialog.error.text())
            creator = service.generate_initial_doe(2, "FACT")
            self.assertEqual(creator.exp_storage.data["exp1"]["T"].iloc[-1], 23.)
            table = ParameterTable([("End (°C)", parameter(25.))], temperature=True)
            table.links[0].setChecked(True)
            table.save()
            self.assertTrue(table.entries[0][1]["equal_previous"])
        finally:
            dialog.close()
            APP.processEvents()

    def test_imported_constant_component_is_defined_once_and_temperature_holds_are_linked(self):
        import pandas as pd
        from experiment.recipe import recipe_from_frame
        service = self.service(template())
        frame = pd.DataFrame({"t": [0., 1., 2.], "T": [25., 24., 24.], "stirring_rate": [300., 300., 300.]})
        recipe = recipe_from_frame(frame, service.exp_par, CONFIGS)
        self.assertTrue(recipe["stages"][0]["actions"]["stirring_rate"]["until_end"])
        self.assertNotIn("stirring_rate", recipe["stages"][1]["actions"])
        self.assertTrue(recipe["stages"][1]["actions"]["T"]["end"]["equal_previous"])
        service.exp_par.timeline = recipe
        compiled = service.generate_initial_doe(1, "LHS").exp_storage.data["exp1"]
        np.testing.assert_allclose(compiled["T"], [25., 24., 24.])
        self.assertTrue(np.all(compiled["stirring_rate"] == 300.))

    def test_shared_plot_style_and_control_center_manual_batch_preview(self):
        from MAIN_WINDOW import MainWindow
        window = MainWindow()
        try:
            configs = {key: deepcopy(CONFIGS[key]) for key in ("TEMP", "PUMP")}
            panel = window.control_panel
            panel.apply_loaded_configuration(configs, {}, [])
            import pandas as pd
            panel.device_widgets["TEMP"].profile = pd.DataFrame({"time_s": [0., 10., 20.], "value": [25., 24., 24.]})
            panel.device_widgets["PUMP"].profile = pd.DataFrame({"time_s": [0., 5.], "value": [1., 0.]})
            for _ in range(2):
                panel.generate_experiment()
            self.assertTrue(panel.view_experiment_profiles_button.isEnabled())
            with patch.object(panel, "connect_all", side_effect=AssertionError("Preview must not connect devices")):
                preview = panel.show_experiment_profiles()
            self.assertEqual(len(preview.canvases), 2)
            manual_axes = preview.canvases[0].figure.axes
            creator = self.service(template()).generate_initial_doe(1, "LHS")
            timeline_axes = creator.plot_doe_experiments()[0].axes
            legacy_axes = ExperimentService().generate_initial_doe(1, "LHS").plot_doe_experiments()[0].axes
            self.assertEqual(manual_axes[0].lines[0].get_color(), legacy_axes[0].lines[0].get_color())
            self.assertEqual(timeline_axes[0].lines[0].get_color(), legacy_axes[0].lines[0].get_color())
            self.assertEqual(timeline_axes[0].get_xlabel(), legacy_axes[0].get_xlabel())
            self.assertEqual(timeline_axes[1].lines[0].get_linestyle(), legacy_axes[1].lines[0].get_linestyle())
            self.assertIn("mL/min", manual_axes[1].get_ylabel())
            self.assertIn("g/h", timeline_axes[1].get_ylabel())
            self.assertEqual(panel.experiments[0]["TEMP"].iloc[-1], 24.)
            preview.close()
        finally:
            window.close()
            APP.processEvents()
