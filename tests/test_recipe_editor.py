"""Recipe editing, exact DoE restore, assignment and timeline execution."""
from copy import deepcopy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QDialog

from app.presets import read_preset, write_preset
from experiment.doe_campaign import DoeCampaignService, ProfileAssignment
from experiment.doe_snapshot import restore_doe, snapshot_doe
from experiment.experiment_service import ExperimentService
from experiment.recipe import parameter
from gui.recipe_editor import COLUMN_WIDTH, HEADER_HEIGHT, LABEL_WIDTH, ROW_HEIGHT, ExperimentDefinitionDialog


APP = QApplication.instance() or QApplication([])


def template():
    lanes = [{"id": key, "role": role, "name": name, "device_id": device, "initial": initial}
             for key, role, name, device, initial in (("T", "T", "Vessel", "TEMP", 25.),
                 ("AS", "AS", "Antisolvent", "PUMP", 0.),
                 ("extra", "clean_sol", "Rinse", "RINSE", 0.), ("valve_state", "valve_state", "Drain", "VALVE", 0.))]
    def action(value, kind="Hold", start=None, mode="Process"):
        return {"name": kind, "kind": kind, "end": parameter(value), "start": parameter(value if start is None else start),
                "control_mode": mode}
    return {"version": 1, "lanes": lanes, "stages": [
        {"id": "charge", "name": "Charge", "purpose": "Charging", "duration": parameter(2.),
         "actions": {"T": action(25., mode="Jacket"), "extra": action(.02, "Dose")}},
        {"id": "cryst", "name": "Cooling and addition", "purpose": "Experiment", "duration": parameter(5.),
         "actions": {"T": action(24., "Ramp", 25.), "AS": action(.03, "Dose")}},
        {"id": "eq", "name": "Equilibrate", "purpose": "Equilibration", "duration": parameter(2.), "actions": {}},
        {"id": "drain", "name": "Discharge", "purpose": "Discharge", "duration": parameter(1.),
         "actions": {"valve_state": action(1.)}},
        {"id": "rinse", "name": "Rinse", "purpose": "Rinse", "duration": parameter(2.),
         "actions": {"extra": action(.01, "Dose")}},
    ]}


CONFIGS = {"TEMP": {"name": "Temperature", "type": "huber", "simulated": True},
           "PUMP": {"name": "AS", "type": "ismatec", "simulated": True},
           "RINSE": {"name": "Rinse", "type": "ismatec", "simulated": True},
           "VALVE": {"name": "Drain", "type": "nidaq_valve", "simulated": True}}


class RecipeEditorTests(unittest.TestCase):
    def service(self):
        service = ExperimentService()
        service.exp_par.v_heat_max = service.exp_par.v_cool_max = 60.
        service.exp_par.timeline = template()
        return service

    def test_sample_alternate_and_compile_custom_lines(self):
        service = self.service()
        service.exp_par.timeline["stages"][1]["actions"]["T"]["end"] = parameter(24., "DoE", 23., 24.)
        service.exp_par.timeline["stages"][1]["actions"]["AS"]["end"] = parameter(.03, "Alternate", .02, .04)
        creator = service.generate_initial_doe(2, "FACT")
        self.assertEqual(creator.actually_sampled_experiments, 2)
        assignments = tuple(ProfileAssignment(lane["id"], lane["device_id"]) for lane in template()["lanes"])
        campaign = DoeCampaignService(creator).compile(assignments, CONFIGS)
        self.assertEqual(len(campaign.experiments[0].profiles), 4)
        self.assertEqual(campaign.experiments[0].model_window, [2., 9.])
        self.assertEqual([item.mode for item in campaign.experiments[0].control_modes], ["Jacket", "Process"])
        finals = [next(p for p in exp.profiles if p.device_id == "TEMP").points[-1].target for exp in campaign.experiments]
        self.assertEqual(set(finals), {23., 24.})
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "recipe.json"
            write_preset(path, {"devices": CONFIGS}, service.exp_par, assignments, {}, {"enabled": True,
                         "sql_enabled": False, "output_folder": folder, "remote_sql_url": ""}, generated_doe=snapshot_doe(creator))
            saved, params, roles = read_preset(path)
            service.exp_par = params
            service._build_doe_dependencies()
            restored = restore_doe(service, saved["generated_doe"])
            second = DoeCampaignService(restored).compile(roles, CONFIGS)
            self.assertEqual(second.experiments[0].control_modes, campaign.experiments[0].control_modes)
            self.assertEqual(second.experiments[0].profiles, campaign.experiments[0].profiles)

    def test_drag_action_and_cancel_preserves_definition(self):
        service = self.service()
        original = deepcopy(service.exp_par.timeline)
        dialog = ExperimentDefinitionDialog(service.exp_par, CONFIGS)
        try:
            editor = dialog.editor
            editor.move_block("charge", "extra", QPointF(LABEL_WIDTH + 2 * COLUMN_WIDTH + 10,
                                                          HEADER_HEIGHT + 2 * ROW_HEIGHT + 10))
            self.assertIn("extra", editor.recipe["stages"][2]["actions"])
            self.assertNotIn("extra", editor.recipe["stages"][0]["actions"])
            with patch.object(QDialog, "exec", return_value=QDialog.DialogCode.Accepted):
                editor.drop_action("Hold", QPointF(LABEL_WIDTH + 10, HEADER_HEIGHT + 2 * ROW_HEIGHT + 10))
            self.assertIn("T", editor.recipe["stages"][2]["actions"])
            dialog.reject()
            self.assertEqual(service.exp_par.timeline, original)
        finally:
            dialog.close()

    def test_editor_main_entry_generates_and_assigns_recipe(self):
        from MAIN_WINDOW import MainWindow
        from gui.doe_assignment_dialog import DoeAssignmentDialog

        window = MainWindow()
        try:
            window.control_panel.apply_loaded_configuration(CONFIGS, {}, [])
            window.experiment_service.exp_par = self.service().exp_par
            def save(dialog):
                dialog.accept()
                return dialog.result()
            with patch.object(ExperimentDefinitionDialog, "exec", save):
                window.open_profile_dialog()
            window.settings_page.exp_number_input.setText("2")
            window.define_doe()
            self.assertEqual(window.create_experiments.actually_sampled_experiments, 2)
            def assign(dialog):
                dialog.preview()
                self.assertIsNotNone(dialog.campaign, dialog.summary.text())
                dialog.accept()
                return dialog.result()
            with patch.object(DoeAssignmentDialog, "exec", assign):
                window.assign_doe_devices()
            self.assertIsNotNone(window._assigned_doe_campaign)
            self.assertIn("extra", window._assigned_doe_campaign.line_roles)
            with TemporaryDirectory(dir=Path.cwd()) as directory:
                service = DoeCampaignService(window.create_experiments)
                folder = service.export_csv(directory, window._assigned_doe_campaign, window._doe_assignments,
                                            window.control_panel.collect_current_configuration()[0])
                self.assertTrue((Path(folder) / "exp1_assigned.csv").is_file())
        finally:
            window.close()
            APP.processEvents()

    def test_component_profile_can_initialize_a_timeline(self):
        service = ExperimentService()
        dialog = ExperimentDefinitionDialog(service.exp_par, CONFIGS)
        try:
            dialog.import_components(CONFIGS)
            self.assertTrue(dialog.editor.recipe["lanes"], dialog.error.text())
            self.assertTrue(dialog.editor.recipe["stages"], dialog.error.text())
            self.assertEqual(dialog.source.currentIndex(), 0)
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted, dialog.error.text())
            self.assertTrue(service.exp_par.timeline)
        finally:
            dialog.close()

    def test_temperature_modes_execute_and_support_survives_modeling(self):
        from experiment.adaptive_loop import RecipeCompiler, default_setup
        from bo.profiles import Profile
        from experiment_dashboard.experiment_dashboard import ExperimentWorker
        from hardware_devices.simulated import SimulatedDevice

        service = self.service()
        creator = service.generate_initial_doe(1, "LHS")
        assignments = tuple(ProfileAssignment(lane["id"], lane["device_id"]) for lane in template()["lanes"])
        campaign = DoeCampaignService(creator).compile(assignments, CONFIGS)
        compiler = RecipeCompiler(default_setup(), campaign, CONFIGS, service.exp_storage, assignments, (), "bo")
        modeled = {"temperature": Profile(np.array([0., 7.]), np.array([24., 24.]), "BO")}
        followup, window = compiler.build(modeled, {"SeedMass": 0}, name="follow-up", source="bo", default_seed=0.)
        self.assertEqual(window, (2., 9.))
        self.assertEqual(len(followup.experiments[0].profiles), 4)
        devices = {key: SimulatedDevice(key, config["type"], noise=0.) for key, config in CONFIGS.items()}
        for device in devices.values():
            device.connect()
        worker = ExperimentWorker(devices, CONFIGS, list(campaign.experiments), speed=100., end_hold_s=0., campaign=campaign)
        with patch.object(devices["TEMP"], "set_process_control", wraps=devices["TEMP"].set_process_control) as process:
            worker.start_experiment()
            self.assertTrue(process.called)
        self.assertEqual(worker.outcome.value, "completed", worker.error_message)

    def test_real_huber_driver_switches_before_stage_setpoints_between_polls(self):
        from experiment.models import ControlModePoint, DeviceProfile, ExperimentDefinition, ProfilePoint
        from experiment_dashboard.experiment_dashboard import ExperimentWorker
        from hardware_devices.hardware_devices import HuberController

        sent = []

        def write(request):
            sent.append(request.decode("ascii").strip())
            return len(request)

        def reply():
            command = sent[-1]
            if "@" in command:
                register, value = command.split("@ ", 1)
                return f"{register} {value}\r\n".encode("ascii")
            return f"{command[:2]} +02200\r\n".encode("ascii")

        huber = HuberController("Huber", "COM1")
        huber.is_connected = True
        huber.ser = Mock(is_open=True)
        huber.ser.write.side_effect = write
        huber.ser.readline.side_effect = reply
        config = {"TEMP": {"type": "huber", "defaults": {"control_mode": "Process"}}}
        temperature = DeviceProfile("TEMP", (ProfilePoint(0., 22.), ProfilePoint(.15, 23.),
                                              ProfilePoint(.4, 24.), ProfilePoint(.55, 24.)), "step", "°C")
        modes = tuple(ControlModePoint(time, "TEMP", mode)
                      for time, mode in ((0., "Jacket"), (.15, "Process"), (.4, "Jacket")))
        experiment = ExperimentDefinition("Huber stage switches", (temperature,), control_modes=modes)
        worker = ExperimentWorker({"TEMP": huber}, config, [experiment], poll_interval=.8, end_hold_s=0.)
        worker.start_experiment()
        self.assertEqual(worker.outcome.value, "completed", worker.error_message)
        self.assertEqual(sent[0], "TM@ 0")
        process = sent.index("TM@ 1")
        jacket = sent.index("TM@ 0", process)
        self.assertLess(process, sent.index("SP@ +02300"))
        self.assertLess(sent.index("SP@ +02300"), jacket)
        self.assertLess(jacket, sent.index("SP@ +02400"))
