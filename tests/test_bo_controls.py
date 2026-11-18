"""Exercise larger BO decision sets through the stage editor and optimizer."""
from dataclasses import asdict
from importlib.util import find_spec
import json
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from bo.physical import Stage, physical_profile
from bo.physical_fit import choose_controls, read_spec
from bo.settings import empty_settings, validate_settings
from bo.stage_editor import StageEditor


APP = QApplication.instance() or QApplication([])


class BOControlTests(unittest.TestCase):
    @staticmethod
    def stages(index):
        temperatures = [60. - stage * (1. + .1 * index) for stage in range(13)]
        rates = [.02 * (1 + stage % 3) + .006 * index for stage in range(12)]
        return {"temperature": tuple(Stage(7., start, end) for start, end in zip(temperatures, temperatures[1:])),
                "feed": tuple(Stage(7., rate, rate) for rate in rates)}

    @unittest.skipUnless(find_spec("sklearn"), "BO requires scikit-learn")
    def test_large_control_sets_round_trip_and_propose(self):
        from bo.optimizer import propose

        template = self.stages(0)
        horizon = 120.
        editor = StageEditor()
        try:
            editor.load({channel: {"stages": [asdict(stage) for stage in stages], "controls": []}
                         for channel, stages in template.items()}, horizon)
            expected = {"temperature": [f"stage:{i}:end" for i in range(12)],
                        "feed": [f"stage:{i}:start" for i in range(12)]}
            for channel, profile_editor in editor.editors.items():
                for row in range(profile_editor.controls.rowCount()):
                    item = profile_editor.controls.item(row, 0)
                    if item.data(Qt.ItemDataRole.UserRole) in expected[channel]:
                        item.setCheckState(Qt.CheckState.Checked)
                self.assertEqual(profile_editor.control_note.text(), "12 decision controls selected.")
            settings = empty_settings()
            settings["physical_profiles"] = editor.specs()
            self.assertGreater(len(choose_controls("temperature", template["temperature"])), 4)
            settings["targets"]["d50"]["enabled"] = True
            for index in range(3):
                profiles = {channel: physical_profile(channel, stages, horizon).payload()
                            for channel, stages in self.stages(index).items()}
                settings["runs"].append({"name": f"Run {index + 1}", "profiles": profiles,
                                         "metrics": {"duration": horizon, "d50": 15. + index,
                                                     **dict.fromkeys(("yield_actual", "yield_theoretical", "d10", "d90"))}})
                settings["selected"].append(index)
            restored = validate_settings(json.loads(json.dumps(settings)))
            editor.load(restored["physical_profiles"], horizon)
            for channel, spec in editor.specs().items():
                self.assertEqual(spec["controls"], expected[channel])
                self.assertEqual(len(read_spec(channel, spec, horizon)[1]), 12)
            result = propose(restored)
            self.assertEqual(len(result.variables), 24)
            for channel in template:
                varied = sum(variable.key.startswith(f"{channel}:") and vary
                             for variable, vary in zip(result.variables, result.vary))
                self.assertGreater(varied, 4)
                self.assertEqual(len(result.stages[channel]), 12)
        finally:
            editor.close()
