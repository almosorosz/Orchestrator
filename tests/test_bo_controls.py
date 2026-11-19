"""Exercise larger BO decision sets through the stage editor and optimizer."""
from dataclasses import asdict
from importlib.util import find_spec
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
import numpy as np

from bo.physical import Stage, apply_controls, physical_profile
from bo.physical_fit import choose_controls, fit_stage_controls, read_spec, trace_spec
from bo.profiles import Profile, make_profile, selected_horizon
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
                if channel not in expected:
                    self.assertEqual(profile_editor.controls.rowCount(), 0)
                    continue
                for row in range(profile_editor.controls.rowCount()):
                    item = profile_editor.controls.item(row, 0)
                    if item.data(Qt.ItemDataRole.UserRole) in expected[channel]:
                        item.setCheckState(Qt.CheckState.Checked)
                self.assertEqual(profile_editor.control_note.text(), "12 decision controls selected.")
            settings = empty_settings()
            settings["batch_horizon_s"] = 125.
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
            self.assertEqual(result.horizon, 125.)
            for channel in template:
                varied = sum(variable.key.startswith(f"{channel}:") and vary
                             for variable, vary in zip(result.variables, result.vary))
                self.assertGreater(varied, 4)
                self.assertEqual(len(result.stages[channel]), 12)
                self.assertEqual(result.profiles[channel].time[-1], 125.)
        finally:
            editor.close()

    def test_use_trace_preserves_long_step_profiles_and_feed_jumps(self):
        horizon = 120.
        step = physical_profile("feed", self.stages(0)["feed"], horizon)
        mixed = physical_profile("feed", (Stage(10., 0., .6), Stage(5., .2, .2), Stage(5., 0., 0.)), horizon)
        for original in (step, mixed):
            with self.subTest(step=original.step):
                spec = trace_spec("feed", original, exact=True)
                stages, _ = read_spec("feed", spec, horizon)
                fitted = physical_profile("feed", stages, horizon)
                self.assertEqual(fitted.step, original.step)
                np.testing.assert_allclose(fitted.sample(original.time), original.values, atol=1e-12)
                self.assertAlmostEqual(fitted.total(), original.total())
        self.assertGreater(len(trace_spec("feed", step, exact=True)["stages"]), 8)

    def test_fits_different_temperatures_in_five_run_imed_preset(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "configurations" / "imed_five_run_control_panel.json"
        experiments = json.loads(path.read_text())["control_panel"]["experiments"]
        profiles = [make_profile(np.asarray([row[:2] for row in experiment["data"] if row[0] <= 60.]).T, "Command · T1")
                    for experiment in experiments]
        spec = trace_spec("temperature", profiles[0], exact=True)
        stages, controls = read_spec("temperature", spec, 60.)
        for profile in profiles:
            values, error = fit_stage_controls("temperature", stages, controls, profile, 60.)
            self.assertLess(error, .01)
            fitted = physical_profile("temperature", apply_controls("temperature", stages, controls, values), 60.)
            self.assertAlmostEqual(fitted.values[-1], profile.values[-1], places=6)

    @unittest.skipUnless(find_spec("sklearn"), "BO requires scikit-learn")
    def test_imed_template_compare_syncs_channel_and_reports_fixed_selected_controls(self):
        from bo.window import BOWindow
        from imed_python.design import _candidate_recipe

        conditions = {"batch": 50., "temperature": [25., 19.], "concentration": .1,
                      "liquid_masses": [100., 0.], "equilibration_s": 10.}
        experiment, process = _candidate_recipe((0., 15., .3, 12., .2), experiment={}, process={},
                                                conditions=conditions, addition=9.6)
        case = {"ExpData": [{**experiment, "Name": f"IMED {i}"} for i in (1, 2)], "ProcInf": [process, process]}
        window = BOWindow(imed_case=lambda: case)
        try:
            window.use_imed()
            editor = window.stage_editor.editors["feed"]
            window.stage_editor.tabs.setCurrentWidget(editor)
            self.assertEqual(window.channel_choice.currentData(), "feed")
            window.use_trace_template()
            original = window.profile(window.state["runs"][0], "feed")
            stages, _ = read_spec("feed", editor.spec(), 60.)
            template = physical_profile("feed", stages, 60.)
            np.testing.assert_allclose(template.sample(original.time), original.values, atol=1e-12)
            self.assertAlmostEqual(template.total(), original.total())

            def run_preview(prepare, complete, **_):
                complete(prepare(lambda _: None))

            with patch.object(window, "run_job", side_effect=run_preview):
                window.preview()
                self.assertEqual(window.fit_summary.rowCount(), 2)
                self.assertEqual(len(window.chart.chart().series()), 5)
                self.assertIn("2 experiments", window.metrics.text())
                for row in range(2):
                    self.assertAlmostEqual(float(window.fit_summary.item(row, 1).text()), 0.)
                window.compare_all.setChecked(False)
                window.preview()
                self.assertEqual(window.fit_summary.rowCount(), 1)
                self.assertEqual(len(window.chart.chart().series()), 3)
            self.assertGreater(window.variables.rowCount(), 0)
            for row in range(window.variables.rowCount()):
                item = window.variables.item(row, 0)
                self.assertEqual(item.text(), "Fixed")
                self.assertIn("Selected in Edit stages", item.toolTip())
            window.channel_choice.setCurrentIndex(window.channel_choice.findData("temperature"))
            temperature = window.stage_editor.editors["temperature"]
            self.assertIs(window.stage_editor.tabs.currentWidget(), temperature)
            original_temperature = temperature.spec()
            temperature.load({"stages": [asdict(Stage(61.9692, 25., 19.))], "controls": []})
            with self.assertRaisesRegex(ValueError, "exceed the fixed batch horizon of 60"):
                window.settings()
            window.batch_horizon.setText("62")
            saved = window.settings()
            self.assertEqual(saved["batch_horizon_s"], 62.)
            self.assertEqual(selected_horizon(saved), 62.)
            window.restore(saved)
            self.assertEqual(float(window.batch_horizon.text()), 62.)
            window.batch_horizon.setText("59")
            with self.assertRaisesRegex(ValueError, "at least 60 s"):
                window.settings()
            temperature.load(original_temperature)
            window.batch_horizon.clear()
            self.assertEqual(selected_horizon(window.settings()), 60.)
        finally:
            window.close()
