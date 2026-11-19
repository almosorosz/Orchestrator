"""Controller basics, discoverable by the standard unittest runner."""
import unittest

from app.controller import Controller, Stage


class ControllerBasicsTests(unittest.TestCase):
    def test_stage_start(self):
        controller = Controller("experiments", "models", 3)
        controller.start()
        self.assertEqual(controller.current_stage, Stage.OMNI)

    def test_expnum(self):
        controller = Controller("experiments", "models", 3)
        self.assertEqual(controller.initial_exp_count, 3)


if __name__ == "__main__":
    unittest.main()
