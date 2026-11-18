import os
from io import StringIO
from unittest.mock import patch
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from bo_app.__main__ import Window
from bo_app.optimizer import load_campaign, rank_candidates


class BayesianAppTest(unittest.TestCase):
    def test_rank_and_handoff_generated_candidate(self):
        app = QApplication.instance() or QApplication([])
        names = ["done1", "done2", "next1", "next2", "pending"]
        profiles = pd.DataFrame([
            {"experiment": name, "time_s": t * (100 + 10 * i),
             "T": 60 - i * t, "AS": t * (i + 1), "stirring_rate": 200}
            for i, name in enumerate(names) for t in (0, 0.5, 1)
        ])
        outcomes = pd.DataFrame({
            "experiment": names, "status": ["completed"] * 2 + ["candidate"] * 2 + ["pending"],
            "seed_amount": [0, 1, 1, 2, 1], "size": [10, 20, None, None, None],
            "c_initial": [1, 1, None, None, None], "c_final": [0.3, None, None, None, None],
            "c_equilibrium": [None, 0.5, None, None, None],
        })
        with patch("bo_app.__main__.load_campaign") as load_for_window:
            inputs = [StringIO(frame.to_csv(index=False)) for frame in (profiles, outcomes)]


            campaign = load_campaign(*inputs)
            self.assertEqual(campaign.features.shape, (5, 10))
            for mode in ("Minimize", "Maximize", "Target"):
                ranking = rank_candidates(campaign, mode, 15, [1, 1, 1], exploration=1)
                self.assertEqual(set(ranking["experiment"]), {"next1", "next2"})
            calls = []
            window = Window(lambda *args: calls.append(args))
            load_for_window.return_value = campaign

            loop = QEventLoop()
            window.run_ranking()
            window.job.finished.connect(loop.quit)
            QTimer.singleShot(60000, loop.quit)
            loop.exec()
            if window.job is not None:
                window.job.wait()
                app.processEvents()
            self.assertTrue(window.start_button.isEnabled(), window.status.text())
            window.start_button.click()
            self.assertEqual(len(calls), 1)
            name, recipe, seed = calls[0]
            pd.testing.assert_frame_equal(
                recipe.reset_index(drop=True),
                profiles.loc[profiles["experiment"] == name].drop(columns="experiment")
                .reset_index(drop=True),
            )
            self.assertEqual(seed, outcomes.set_index("experiment").loc[name, "seed_amount"])
            self.assertFalse(window.start_button.isEnabled())


if __name__ == "__main__":
    unittest.main()

