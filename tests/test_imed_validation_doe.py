import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from app.presets import read_preset
from experiment.doe_campaign import DoeCampaignService
from experiment.doe_snapshot import restore_doe
from experiment.experiment_service import ExperimentService
from experiment_dashboard.experiment_dashboard import ExperimentWorker
from hardware_devices.simulated import SimulatedDevice
from imed_python.demo_campaign import import_simulated_campaign
from imed_python.estimation import FitObjective
from imed_python.solver import model_parameters


class ImedValidationDoeTests(unittest.TestCase):
    def test_recorded_assignments_and_crystallization_window_round_trip(self):
        app = QApplication.instance() or QApplication([])
        preset = Path(__file__).resolve().parents[1] / "examples/configurations/imed_validation_doe.json"
        data, parameters, assignments = read_preset(preset)
        configs = data["control_panel"]["devices"]
        configs["renamed_concentration"] = configs.pop("IR1")
        monitor_ids = [key for key in data["monitor_device_ids"] if key != "IR1"]
        monitor_ids.append("renamed_concentration")
        service = ExperimentService()
        service.exp_par = parameters
        service._build_doe_dependencies()
        creator = restore_doe(service, data["generated_doe"])
        doe = DoeCampaignService(creator)
        monitors = {**doe.imed_simulations["exp1"]["monitors"], "Concentration": "renamed_concentration"}
        campaign = doe.compile(assignments, configs, monitor_ids, imed_monitors=monitors)
        devices = {key: SimulatedDevice(config["name"], config["type"], noise=0)
                   for key, config in configs.items()}
        for device in devices.values():
            device.connect()
        with TemporaryDirectory(dir=Path.cwd()) as output, patch("app.campaign_roots.register_root"):
            worker = ExperimentWorker(devices, configs, campaign.experiments, speed=100,
                                      campaign=campaign, poll_interval=2,
                                      recording={"enabled": True, "output_folder": output})
            worker.start_experiment()
            self.assertEqual(worker.outcome.value, "completed", worker.error_message)
            case = import_simulated_campaign(worker.recorder.folder)
            self.assertEqual(len(case["ExpData"]), 3)
            for exp, proc, run in zip(case["ExpData"], case["ProcInf"], worker.runs):
                time, temp = exp["TemperatureData"]
                self.assertEqual(time[0], 0)
                self.assertAlmostEqual(time[-1], 60)
                np.testing.assert_allclose(temp[time >= 35], temp[-1])
                self.assertLess(exp["ConcentrationData"][1, -1], exp["InitialConcentration"])
                self.assertTrue(np.all(np.diff(exp["FinalD10D50D90"]) > 0))
                self.assertGreater(np.ptp(exp["CountsData"][1]), 0)
                self.assertEqual(bool(exp["CountsData"][1, 0]), bool(proc["SeedMass"]))
                flow = exp["FlowRateData"]
                self.assertAlmostEqual(np.trapezoid(flow[1], flow[0]), exp["AddedAntisolventData"][1, -1])
                csv = pd.read_csv(worker.recorder.folder / run.run_id / "execution.csv")
                self.assertEqual(csv.loc[csv.elapsed_s > 60, "V1.target_state"].max(), 1)
                self.assertGreater(csv.loc[csv.elapsed_s > 60, "PCLEN1.target"].max(), 0)
                self.assertTrue(csv.loc[csv.elapsed_s > 60, "renamed_concentration.value"].isna().all())
        objective = FitObjective(case["ExpData"], case["ProcInf"], case["Solubility"],
                                 case["pbmsetting"], [0, 1, 0], "D10 D50 D90")
        self.assertTrue(np.isfinite(objective.residuals(model_parameters(case["ModelParameter"][0]))).all())
