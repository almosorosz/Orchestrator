from pathlib import Path
from dataclasses import asdict
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from bo.profiles import command_profiles
from experiment.doe_campaign import DoeCampaignService, ProfileAssignment, convert_profile
from experiment.experiment_service import ExperimentService
from experiment.material_balance import pump_mass
from experiment.models import DeviceProfile, ExperimentDefinition, ProfilePoint, RunStatus
from experiment.pump_calibration_files import calibration_knots, load_calibration, load_seed_calibration, seed_dose_command
from experiment.pump_corrections import apply_pump_corrections
from experiment.recipe import parameter
from imed_python.campaign_import import import_run
from test_doe_campaign import fixture_configs


def calibrated_ismatec():
    return {"type": "ismatec", "reservoir_density_g_ml": .73,
            "pump_calibration": {"version": 1, "input_unit": "g/h", "output_unit": "RPM",
                                 "source_name": "measured.csv", "points": [[0., 0.], [720., 12.], [1440., 48.]]}}


class PumpCorrectionTests(unittest.TestCase):
    def test_seed_curve_in_minutes_and_real_ismatec_calibrates_without_density(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            file = Path(folder) / "seed.csv"
            file.write_text("time_min,dry_g\n0,0\n1,0.12\n2,0.30\n", encoding="utf-8")
            cfg = {"type": "masterflex"}
            cfg["seed_calibration"] = load_seed_calibration(file, cfg, "time_min", "dry_g", 50., time_unit="min")
            rpm, seconds = seed_dose_command(cfg, None, .21)
            self.assertEqual(rpm, 50.)
            self.assertAlmostEqual(seconds, 90.)
            self.assertEqual(seed_dose_command(cfg, None, 0.), (50., 0.))
            with self.assertRaisesRegex(ValueError, "calibrated range"):
                seed_dose_command(cfg, None, .31)
            file.write_text("mass_g_min,rpm\n12,12\n24,48\n", encoding="utf-8")
            ism = {"type": "ismatec", "reservoir_density_g_ml": .73}
            as_assignment = ProfileAssignment("AS", "P")
            with self.assertRaisesRegex(ValueError, "antisolvent mass-flow calibration"):
                convert_profile(None, as_assignment, ism, np.array([720.]))
            ism["pump_calibration"] = load_calibration(file, "ismatec", "mass_g_min", "rpm", flow_unit="g/min")
            commands, unit, _ = convert_profile(None, as_assignment, ism, np.array([720., 1080.]))
            np.testing.assert_allclose(commands, [12., 30.])
            self.assertEqual(unit, "RPM")
            times, rates = calibration_knots(ism, np.array([0., 10.]), np.array([0., 1440.]))
            np.testing.assert_allclose(times, [0., 5., 10.])
            np.testing.assert_allclose(rates, [0., 720., 1440.])

    def test_prime_and_reverse_are_timed_once_and_reserve_original_process_feed(self):
        cfg = calibrated_ismatec()
        cfg["pump_corrections"] = {"prime_g": 1., "reverse_target": 6., "reverse_s": 6.}
        feed = DeviceProfile("P", (ProfilePoint(0., 0.), ProfilePoint(2., 12.), ProfilePoint(20., 0.)), "step", "RPM")
        temp = DeviceProfile("T", (ProfilePoint(0., 25.), ProfilePoint(20., 25.)), unit="°C")
        experiment = ExperimentDefinition("Batch", (feed, temp), model_window=(0., 20.))
        planned = apply_pump_corrections(experiment, {"P": cfg, "T": {"type": "huber"}})
        self.assertIs(apply_pump_corrections(planned, {"P": cfg}), planned)
        self.assertEqual(planned.model_window, (3., 23.))
        actions = planned.pump_adjustments["pumps"]["P"]
        self.assertEqual((actions["prime"]["start_s"], actions["prime"]["end_s"]), (0., 5.))
        self.assertEqual((actions["reverse"]["start_s"], actions["reverse"]["end_s"]), (23., 29.))
        self.assertAlmostEqual(pump_mass(planned.profiles[0], cfg), 1. + 3.6 - .6)
        profiles = command_profiles(asdict(planned), {}, {"devices": {"P": cfg}}, 3., 20., {"AS": "P"})
        self.assertAlmostEqual(profiles["feed"].total(), 3.6)

    def test_simulated_worker_executes_and_records_back_rotation(self):
        from control_panel.control_panel import HARDWARE_TYPES, create_hardware
        from experiment_dashboard.experiment_dashboard import ExperimentWorker
        from imed_python.campaign_import import campaign_runs
        import csv

        configs = {"PMP_AS": {**fixture_configs()["PMP_AS"], **calibrated_ismatec()}}
        configs["PMP_AS"]["pump_corrections"] = {"prime_g": .2, "reverse_target": 12., "reverse_s": 1.}
        feed = DeviceProfile("PMP_AS", (ProfilePoint(0., 12.), ProfilePoint(1., 0.)), "step", "RPM")
        definition = apply_pump_corrections(ExperimentDefinition("Test", (feed,)), configs)
        hardware = create_hardware(configs, HARDWARE_TYPES)
        hardware["PMP_AS"].connect()
        self.addCleanup(hardware["PMP_AS"].disconnect)
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            worker = ExperimentWorker(hardware, configs, [definition], speed=100., end_hold_s=0.,
                                      recording={"enabled": True, "output_folder": folder})
            worker.start_experiment()
            self.assertEqual(worker.outcome, RunStatus.COMPLETED, worker.error_message)
            campaign = next(Path(folder).glob("campaign_*"))
            _, _, file = campaign_runs(campaign)[0]
            with file.open(newline="", encoding="utf-8-sig") as stream:
                rows = {float(row["elapsed_s"]): float(row["PMP_AS.target"]) for row in csv.DictReader(stream)}
            self.assertEqual(rows[0.], 12.)
            self.assertEqual(rows[1.], 12.)
            self.assertEqual(rows[2.], -12.)
            self.assertEqual(rows[3.], 0.)

    def test_scale_import_retains_only_reactor_dose_when_prime_overlaps_window(self):
        action = {"start_s": 0., "end_s": 5., "mass_g": 1.}
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            file = Path(folder) / "execution.csv"
            file.write_text("elapsed_s,T,S\n3,25,99.4\n5,25,99\n13,25,97.4\n23,25,95.4\n", encoding="utf-8")
            result = import_run(file, {"Temperature": "T", "AS scale": "S"}, start=3., end=23.,
                concentration_multiplier=1., scale_direction=-1, smoothing=1, initial_concentration=.1,
                seed_percentiles=None, no_feed=False, pump_adjustment={"prime": action})
            self.assertAlmostEqual(result["AddedAntisolventData"][1, -1], 3.6)
            self.assertAlmostEqual(result["AntisolventScaleData"][1, -1], 95.4)

    def test_dry_seed_mass_action_resolves_to_pulse_at_assignment(self):
        service = ExperimentService()
        amount = .21
        service.exp_par.timeline = {"version": 1, "lanes": [
            {"id": "seed", "role": "seed", "name": "Seed", "initial": 0.},
            {"id": "T", "role": "T", "name": "Temperature", "initial": 25.}], "stages": [
            {"id": "charge", "name": "Seed addition", "purpose": "Seeding", "duration": parameter(100.),
             "actions": {"seed": {"kind": "Dose", "end": parameter(amount)}}},
            {"id": "batch", "name": "Experiment", "purpose": "Experiment", "duration": parameter(50.), "actions": {}}]}
        creator = service.generate_initial_doe(1, "LHS")
        configs = fixture_configs()
        configs["PMP_SEED"]["seed_calibration"] = {"version": 1, "rpm": 50.,
                                                  "points": [[0., 0.], [60., .12], [120., .30]]}
        campaign = DoeCampaignService(creator).compile((ProfileAssignment("seed", "PMP_SEED"),
                                                        ProfileAssignment("T", "TEM_MAIN")), configs)
        run = campaign.experiments[0]
        points = run.to_frame()
        self.assertAlmostEqual(run.seed_mass_g, amount)
        self.assertAlmostEqual(points.loc[points["PMP_SEED"] > 0, "time_s"].max(), 90., places=5)
        self.assertEqual(tuple(run.model_window), (100., 150.))
