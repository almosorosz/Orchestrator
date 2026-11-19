"""All-device inheritance, draft previews and crowded recipe layouts."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QPlainTextEdit, QDialog

from app.presets import read_preset, write_preset
from experiment.doe_campaign import ROLES
from experiment.experiment_service import ExperimentService
from experiment.recipe import parameter
from gui.device_profile_dialog import DeviceProfileDialog
from gui.recipe_editor import COLUMN_WIDTH, HEADER_HEIGHT, LABEL_WIDTH, ROW_HEIGHT, ExperimentDefinitionDialog
from test_recipe_editor import APP, CONFIGS


def target(value, kind="Hold", linked=False):
    result = {"name": kind, "kind": kind, "control_mode": "Process", "start": parameter(value), "end": parameter(value)}
    result["end"]["equal_previous"] = linked
    return result


def all_devices():
    lanes, first, second = [], {}, {}
    for role in ROLES:
        initial = 25. if role == "T" else 0.
        lanes.append({"id": role, "name": ROLES[role].label, "role": role, "initial": initial, "device_id": ""})
        low, high = (0., 1.) if role == "valve_state" else (23., 24.) if role == "T" else (6., 12.)
        first[role] = target(low)
        first[role]["end"] = parameter(low, "Alternate", low, high)
        second[role] = target(0., linked=True)
        # Hidden stale sampling settings must not add dimensions or override links.
        second[role]["end"].update(mode="DoE", min=0., max=1.)
    return {"version": 1, "time_unit": "min", "lanes": lanes, "stages": [
        {"id": "first", "name": "First", "purpose": "Experiment", "duration": parameter(60.), "actions": first},
        {"id": "second", "name": "Second", "purpose": "Equilibration", "duration": parameter(60.), "actions": second}]}


class RecipePreviewTests(unittest.TestCase):
    def service(self, recipe=None):
        service = ExperimentService()
        service.exp_par.timeline = recipe or all_devices()
        service.exp_par.v_heat_max = service.exp_par.v_cool_max = 60.
        return service

    def test_shared_links_resolve_per_experiment_for_every_supported_device_and_reload(self):
        service = self.service()
        creator = service.generate_initial_doe(2, "LHS")
        self.assertEqual(creator.dimensions, 0)
        for name, frame in creator.exp_storage.data.items():
            for role in ROLES:
                with self.subTest(experiment=name, role=role):
                    self.assertTrue(np.all(frame[role] == frame[role].iloc[0]))
                    self.assertEqual(creator.parameter_snapshots[name][f"second.{role}.end"]["value"], frame[role].iloc[0])
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "all-devices.json"
            write_preset(path, {"devices": CONFIGS}, service.exp_par, (), {},
                         {"enabled": True, "sql_enabled": False, "output_folder": folder})
            _, parameters, _ = read_preset(path)
            for action in parameters.timeline["stages"][1]["actions"].values():
                self.assertTrue(action["end"]["equal_previous"])

    def test_independent_linked_mass_addition_keeps_pump_flow_and_derives_mass(self):
        recipe = {"version": 1, "lanes": [{"id": "AS", "name": "AS", "role": "AS", "initial": 0., "profile": [
            {"id": "feed", "duration": parameter(60.), **target(12.)},
            {"id": "dose", "duration": parameter(30.), **target(0., "Dose", linked=True)}]}], "stages": []}
        creator = self.service(recipe).generate_initial_doe(1, "LHS")
        frame = creator.exp_storage.data["exp1"]
        self.assertAlmostEqual(creator.parameter_snapshots["exp1"]["profile:AS:dose.AS.end"]["value"], .1)
        self.assertAlmostEqual(np.interp(75 / 3600, frame["t"], frame["AS"]), 12.)
        self.assertAlmostEqual(np.trapezoid(frame["AS"], frame["t"]), .3, places=7)
        recipe["lanes"][0]["profile"][0]["end"] = parameter(-12.)
        with self.assertRaisesRegex(ValueError, "negative pump flow"):
            self.service(recipe).generate_initial_doe(1, "LHS")

    def test_other_device_profile_editor_offers_previous_for_target_but_not_duration(self):
        for role in ROLES:
            with self.subTest(role=role):
                lane = next(lane for lane in all_devices()["lanes"] if lane["role"] == role)
                dialog = DeviceProfileDialog(lane, [])
                try:
                    self.assertNotIn(0, dialog.table.links)
                    self.assertIn(1, dialog.table.links)
                    dialog.table.links[1].setChecked(True)
                    dialog.accept()
                    self.assertTrue(dialog.sections[0]["end"]["equal_previous"])
                finally:
                    dialog.close()

    def test_representative_uses_unsaved_values_and_links_without_generating_or_saving(self):
        service = self.service()
        original = asdict(service.exp_par)
        dialog = ExperimentDefinitionDialog(service.exp_par, CONFIGS)
        try:
            cooling = dialog.editor.recipe["stages"][0]["actions"]["T"]["end"]
            cooling.update(mode="DoE", value=22., min=10., max=30.)
            with patch.object(ExperimentService, "generate_initial_doe", side_effect=AssertionError("Preview must compile just the draft")):
                preview = dialog.preview_experiment()
            self.assertIsNotNone(preview, dialog.error.text())
            self.assertEqual(len(preview.canvases), 1)
            axes = preview.canvases[0].figure.axes
            self.assertTrue(np.all(axes[0].lines[0].get_ydata() == 22.))
            self.assertEqual(axes[0].get_xlabel(), "Time (min)")
            self.assertEqual(asdict(service.exp_par), original)
            # Current edits are picked up by a second click without re-saving.
            cooling["value"] = 21.
            next_preview = dialog.preview_experiment()
            self.assertTrue(np.all(next_preview.canvases[0].figure.axes[0].lines[0].get_ydata() == 21.))
            dialog.reject()
            self.assertIsNone(dialog.preview_dialog)
            self.assertEqual(asdict(service.exp_par), original)
        finally:
            dialog.close()
            APP.processEvents()

    def test_preview_plots_rate_limited_draft_and_keeps_invalid_structure_editable(self):
        recipe = all_devices()
        recipe["lanes"] = recipe["lanes"][:1]
        for stage in recipe["stages"]:
            stage["actions"] = {"T": stage["actions"]["T"]}
        recipe["stages"][0]["actions"]["T"] = target(15., "Ramp")
        recipe["stages"][0]["actions"]["T"]["start"] = parameter(25.)
        recipe["stages"][0]["duration"] = parameter(10.)
        service = self.service(recipe)
        service.exp_par.v_cool_max = 1.
        dialog = ExperimentDefinitionDialog(service.exp_par, CONFIGS)
        try:
            self.assertIsNotNone(dialog.preview_experiment(), dialog.error.text())
            self.assertTrue(any("Minimum duration" in text.toPlainText() for text in dialog.preview_dialog.findChildren(QPlainTextEdit)))
            dialog.editor.recipe["stages"][0]["duration"]["value"] = 0.
            self.assertIsNone(dialog.preview_experiment())
            self.assertIn("positive", dialog.error.text())
            self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        finally:
            dialog.reject()
            APP.processEvents()

    def test_many_stages_and_short_independent_sections_get_separate_space(self):
        recipe = all_devices()
        recipe["lanes"] = recipe["lanes"][:2]
        base = deepcopy(recipe["stages"][0])
        recipe["stages"] = []
        for index in range(35):
            stage = deepcopy(base)
            stage.update(id=f"stage{index}", name=f"Long stage {index}")
            stage["actions"] = {"T": stage["actions"]["T"]}
            recipe["stages"].append(stage)
        lane = recipe["lanes"][1]
        lane["profile"] = [{"id": f"short{i}", "duration": parameter(1.), **target(float(i))} for i in range(12)]
        dialog = ExperimentDefinitionDialog(self.service(recipe).exp_par, CONFIGS)
        try:
            dialog.resize(1050, 700)
            dialog.show()
            APP.processEvents()
            editor = dialog.editor
            self.assertEqual(editor.stage_list.count(), 35)
            self.assertGreater(editor.view.verticalScrollBar().maximum(), 0)
            self.assertGreater(editor.sidebar.verticalScrollBar().maximum(), 0)
            self.assertGreaterEqual(editor.stage_edges[1] - editor.stage_edges[0], ROW_HEIGHT * 12)
            # Drag/drop coordinates still refer to shared stages after expansion.
            from PySide6.QtCore import QPointF
            point = QPointF(LABEL_WIDTH + COLUMN_WIDTH / 2, editor.stage_edges[2] + 10)
            self.assertEqual(editor.cell(point), (2, 0))
        finally:
            dialog.close()
            APP.processEvents()
