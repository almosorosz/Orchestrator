"""Time displays convert without changing executable profile timing."""
from copy import deepcopy
import unittest

from PySide6.QtWidgets import QDialog

from experiment.experiment_service import ExperimentService
from experiment.recipe import parameter
from gui.device_profile_dialog import DeviceProfileDialog
from gui.recipe_editor import ExperimentDefinitionDialog, ParameterTable
from test_recipe_editor import APP, CONFIGS, template


class RecipeTimeUnitTests(unittest.TestCase):
    def service(self):
        service = ExperimentService()
        recipe = template()
        recipe["lanes"] = recipe["lanes"][:1]
        recipe["stages"] = [recipe["stages"][1]]
        recipe["stages"][0]["duration"] = parameter(60.)
        recipe["stages"][0]["actions"].pop("AS")
        service.exp_par.timeline = recipe
        service.exp_par.v_cool_max = 1.
        service.exp_par.v_heat_max = 2.
        return service

    def test_duration_fields_convert_values_bounds_and_repeated_edits(self):
        duration = parameter(60., "DoE", 30., 120.)
        table = ParameterTable([("Duration (s)", duration)], time_unit="min")
        self.assertEqual(table.item(0, 0).text(), "Duration (min)")
        self.assertEqual([table.cellWidget(0, col).value() for col in (2, 3, 4)], [1., .5, 2.])
        table.cellWidget(0, 2).setValue(2.)
        table.save()
        self.assertEqual(duration["value"], 120.)
        table.cellWidget(0, 2).setValue(1.)
        table.save()
        self.assertEqual(duration["value"], 60.)
        for unit in ("s", "min", "h"):
            with self.subTest(unit=unit):
                original = parameter(1., "Alternate", .123456789, 3.123456789)
                item = deepcopy(original)
                ParameterTable([("Duration (s)", item)], time_unit=unit).save()
                self.assertEqual(item, original)

    def test_switching_units_preserves_actual_rates_and_profile_duration(self):
        service = self.service()
        dialog = ExperimentDefinitionDialog(service.exp_par, CONFIGS)
        try:
            self.assertEqual(dialog.time_units.currentData(), "min")
            self.assertEqual(dialog.system_labels["v_cool_max"].text(), "Maximum cooling rate (°C/min)")
            for unit in ("s", "h", "min"):
                dialog.time_units.setCurrentIndex(dialog.time_units.findData(unit))
                self.assertEqual(dialog.rate_value("v_cool_max"), 1.)
            dialog.time_units.setCurrentIndex(dialog.time_units.findData("s"))
            self.assertAlmostEqual(dialog.system["v_cool_max"].value(), 1 / 60, places=8)
            dialog.system["v_cool_max"].setValue(.1)
            dialog.time_units.setCurrentIndex(dialog.time_units.findData("h"))
            self.assertEqual(dialog.system["v_cool_max"].value(), 360.)
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted, dialog.error.text())
            self.assertEqual(service.exp_par.v_cool_max, 6.)
            self.assertEqual(service.exp_par.timeline["time_unit"], "h")
            creator = service.generate_initial_doe(1, "LHS")
            self.assertEqual(creator.exp_storage.data["exp1"]["t"].iloc[-1] * 3600, 60.)
        finally:
            dialog.close()
            APP.processEvents()

    def test_save_allows_rate_limited_draft_but_generation_reports_required_duration(self):
        service = self.service()
        service.exp_par.timeline["stages"][0]["duration"] = parameter(30.)
        dialog = ExperimentDefinitionDialog(service.exp_par, CONFIGS)
        try:
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted, dialog.error.text())
            self.assertEqual(len(dialog.limit_warnings), 1)
            self.assertIn("requires 2 °C/min", dialog.limit_warnings[0])
            self.assertIn("Minimum duration: 1 min (60 s)", dialog.limit_warnings[0])
            with self.assertRaisesRegex(ValueError, "Minimum duration: 1 min"):
                service.generate_initial_doe(1, "LHS")
            service.exp_par.timeline["stages"][0]["duration"] = parameter(60.)
            creator = service.generate_initial_doe(1, "LHS")
            self.assertEqual(creator.exp_storage.data["exp1"]["T"].iloc[-1], 24.)
        finally:
            dialog.close()
            APP.processEvents()

    def test_independent_section_uses_chosen_units_without_retiming(self):
        lane = deepcopy(self.service().exp_par.timeline["lanes"][0])
        lane["profile"] = [{"id": "cool", "name": "Cooling", "kind": "Ramp", "control_mode": "Process",
                            "duration": parameter(120.), "start": parameter(25.), "end": parameter(24.)}]
        dialog = DeviceProfileDialog(lane, [], time_unit="min")
        try:
            self.assertEqual(dialog.table.item(0, 0).text(), "Duration (min)")
            self.assertEqual(dialog.table.cellWidget(0, 2).value(), 2.)
            self.assertIn("2 min", dialog.section_list.item(0).text())
            dialog.accept()
            self.assertEqual(dialog.sections[0]["duration"]["value"], 120.)
        finally:
            dialog.close()

    def test_invalid_duration_still_blocks_save_and_legacy_source_stays_selected(self):
        service = self.service()
        service.exp_par.timeline["stages"][0]["duration"] = parameter(0.)
        dialog = ExperimentDefinitionDialog(service.exp_par, CONFIGS)
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        self.assertIn("positive", dialog.error.text())
        dialog.close()
        legacy = ExperimentDefinitionDialog(ExperimentService().exp_par, CONFIGS)
        try:
            self.assertEqual(legacy.source.currentText(), "Component design")
            legacy.time_units.setCurrentIndex(legacy.time_units.findData("s"))
            self.assertEqual(legacy.source.currentText(), "Component design")
        finally:
            legacy.close()
