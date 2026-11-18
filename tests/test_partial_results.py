"""Stopping modeling retains completed evaluations and their saved history."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from imed_python.design import optimize_recipe
from imed_python.estimation import fit_model
from imed_python.estimation_dialog import ParameterEstimation
from imed_python.native_case import read_native_case, save_native_case
from imed_python.probabilistic import propagate
from imed_python.session import latest_pe_step, unpack_fit
from imed_python.test_probabilistic import ProbabilisticTests


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APP = QApplication.instance() or QApplication([])


class SimpleObjective:
    residual_count = sample_count = 8

    def residuals(self, parameters, *, global_stage=False):
        return np.arange(1., 9.) * (parameters[0] - .33)


class PartialResultTests(unittest.TestCase):
    def test_stopped_global_fit_retains_best_and_reopens(self):
        stop = Event()
        paraset = np.zeros((35, 4))
        paraset[0] = [1., 1., -2., 2.]
        fit = fit_model({"paraset": paraset}, SimpleObjective(), global_iterations=20, population_size=6,
                        cancelled=stop.is_set, generation_progress=lambda *args: stop.set(), retain_partial=True)
        self.assertEqual(fit.stopped_stage, "Differential evolution")
        self.assertLess(fit.ssr, float(np.sum(SimpleObjective().residuals([1.])**2)))
        case = {"ModelParameter": [{"paraset": paraset}], "ExpData": [{}], "ProcInf": [{}]}
        dialog = ParameterEstimation(case)
        try:
            dialog.show_results({0: fit}, {})
            dialog.job = SimpleNamespace(cancel_requested=stop, deleteLater=lambda: None)
            dialog.finish_fit()
            self.assertIs(dialog.results[0], fit)
            self.assertTrue(dialog.save_button.isEnabled())
            with TemporaryDirectory(dir=Path.cwd()) as directory:
                path = save_native_case(Path(directory) / "stopped.imed.json", dialog.case)
                saved = read_native_case(path)
                restored = unpack_fit(latest_pe_step(saved, 0)["result"])
                np.testing.assert_array_equal(restored.parameters, fit.parameters)
                self.assertEqual(restored.stopped_stage, fit.stopped_stage)
        finally:
            dialog.close()

    def test_stopped_local_fit_retains_evaluated_parameters(self):
        stop = Event()
        class Objective(SimpleObjective):
            calls = 0
            def residuals(self, parameters, *, global_stage=False):
                self.calls += 1
                if self.calls == 3:
                    stop.set()
                return super().residuals(parameters, global_stage=global_stage)
        paraset = np.zeros((35, 4))
        paraset[0] = [1., 1., -2., 2.]
        fit = fit_model({"paraset": paraset}, Objective(), global_iterations=0,
                        cancelled=stop.is_set, retain_partial=True)
        self.assertEqual(fit.stopped_stage, "Local least squares")
        self.assertLess(abs(fit.parameters[0] - .33), abs(1. - .33))

    def test_stopped_design_keeps_best_recipe(self):
        stop = Event()
        scores = []
        conditions = {"batch": 30, "temperature": (20, 15), "temperature_rates": (.5, 1),
                      "liquid_masses": (10, 0), "concentration": .1, "final_anti": .2,
                      "antisolvent_rates": (.2, .5), "seed_bounds": (0, 1)}
        def score(experiment, process):
            value = experiment["TemperatureData"][0, 1] - process["SeedMass"]
            scores.append(value)
            if len(scores) == 8:
                stop.set()
            return value
        value, _, _ = optimize_recipe({}, {}, conditions, score, iterations=20,
                                       cancelled=stop.is_set, retain_partial=True)
        self.assertEqual(value, max(scores))

    def test_stopped_analysis_uses_only_finished_draws(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):
                stop = Event()
                result = propagate(ProbabilisticTests.case(), 0, 0, samples=20, workers=workers,
                                   cancelled=stop.is_set, retain_partial=True,
                                   draw_progress=lambda done, total, success: stop.set() if done == 3 else None)
                self.assertTrue(result["stopped"])
                self.assertGreaterEqual(len(result["valid"]), 3)
                self.assertLess(len(result["valid"]), 20)
                self.assertEqual(result["inputs"].shape[0], len(result["valid"]))
                self.assertEqual(result["predictions"].shape[0], np.count_nonzero(result["valid"]))
