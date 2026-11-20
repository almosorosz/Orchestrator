"""Create simulated Control Panel loop presets from saved pump calibrations."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.presets import write_preset
from experiment.profiles.profile_gen import ExperimentProfileParameters
from experiment.pump_calibration_files import convert_flow, seed_dose_command, validate_calibration, validate_seed_calibration
from experiment.run_stages import RunStage
from imed_python.demo_campaign import model_inputs, validation_model
from imed_python.setup_snapshot import pack_imed_settings


CONTROLS = {"T": "T1", "AS": "PAS1", "solution": "PSOLU1", "solvent": "PSOLV1", "seed": "PSEED1",
            "clean_sol": "PCLEN1", "stirring_rate": "SMAIN1", "valve_state": "V1"}
MONITORS = {"Concentration": "IR1", "FBRM": "FBRM1", "D10": "D101", "D50": "D501", "D90": "D901", "AS scale": "WAS1"}
STAGES = tuple(RunStage(*data) for data in (
    ("charge", "Charge the reactor", "Charging", 0., 45.),
    ("seed", "Dose dry seed", "Seeding", 45., 62.),
    ("crystallize", "Cool and add antisolvent", "Experiment", 62., 132.),
    ("hold", "Equilibrate", "Equilibration", 132., 162.),
    ("drain", "Drain reactor", "Discharge", 162., 168.),
    ("clean", "Rinse and drain", "Cleaning", 168., 184.),
))


def starting_frames(devices):
    def charge_rpm(device, grams, seconds):
        config = devices[device]
        return float(convert_flow(config["pump_calibration"], config["type"], grams * 3600. / seconds))

    for factor, final, feed, seed_mass in zip((.90, 1., 1.10, .95, 1.05), (15., 19., 17., 21., 13.),
                                               (8., 14., 18., 5., 11.), (.025, .055, .04, .085, .07)):
        rpm, seed_seconds = seed_dose_command(devices["PSEED1"], None, seed_mass)
        def pulse(start, end, value):
            # Control Panel saves linear profiles. The adjacent knots preserve the pulse edges.
            return {max(0., start - 1e-6): 0., start: value, end - 1e-6: value, end: 0.}

        profiles = {"T1": {0.: 25., 62.: 25., 132.: final, 184.: final}, "SMAIN1": {0.: 200.},
                    "PSOLU1": pulse(0., 40., charge_rpm("PSOLU1", 50.*factor, 40.)),
                    "PSOLV1": pulse(0., 45., charge_rpm("PSOLV1", 23.*factor, 45.)),
                    "PSEED1": pulse(45., 45. + seed_seconds, rpm),
                    "PAS1": {**pulse(5., 40., 9.), **pulse(72., 127., feed)},
                    "PCLEN1": pulse(169., 176., 25.), "V1": {0.: 0., 163.: 1., 167.: 0., 178.: 1., 183.: 0.}}
        times = sorted({0., 184., *(stage.start_s for stage in STAGES), *(stage.end_s for stage in STAGES),
                        *(time for profile in profiles.values() for time in profile)})
        columns = {}
        for device, points in profiles.items():
            if device == "V1":
                x, y = zip(*sorted(points.items()))
                columns[device] = np.asarray(y)[np.searchsorted(x, times, side="right") - 1]
            else:
                columns[device] = np.interp(times, *zip(*sorted(points.items())))
        yield pd.DataFrame({"time_s": times, **columns}).to_dict(orient="split")


def generate(output, calibration_config):
    examples = Path(__file__).resolve().parents[1] / "examples" / "configurations"
    base = json.loads((examples / "crystallization_system_simulated.json").read_text(encoding="utf-8"))
    saved = json.loads(calibration_config.read_text(encoding="utf-8-sig"))["devices"]
    devices = {key: base["devices"][key] if key in base["devices"] else saved[key]
               for key in (*CONTROLS.values(), *MONITORS.values())}
    for key, config in devices.items():
        config["simulated"] = True
        config.setdefault("simulation", {})["noise"] = 0.
        if key in {"PAS1", "PSOLU1", "PSOLV1", "PCLEN1"}:
            config["pump_calibration"] = saved[key]["pump_calibration"]
            validate_calibration(config["pump_calibration"], config["type"])
    devices["PSEED1"]["seed_calibration"] = saved["PSEED1"]["seed_calibration"]
    validate_seed_calibration(devices["PSEED1"]["seed_calibration"], devices["PSEED1"])
    nominal, _, process, solubility, mesh = model_inputs(True)
    process = {key: value for key, value in process.items() if key not in {"SolvMass", "AntiSolvMass", "SeedMass"}}
    setup = {"max_followups": 1, "manual_assignments": CONTROLS, "derive_initial_charges": True,
             "manual_simulation": {"hold_start_s": 70., "end_s": 100., "monitors": MONITORS},
             "recording": {"initial_concentration": .1, "seed_percentiles": [6., 10., 15.], "smoothing": 1},
             "process": process, "imed": {"iterations": 0, "covariance_samples": 2, "delta_aic": 6}}
    stages = [asdict(stage) for stage in STAGES]
    frames = list(starting_frames(devices))
    panel = {"devices": devices, "profiles": {}, "experiments": frames, "simulation_speed": 100.,
             "experiment_metadata": [{"run_stages": stages} for _ in frames], "system_topology": base["system_topology"]}
    output.mkdir(parents=True, exist_ok=True)
    for backend, suffix in (("bo", "bo"), ("imed", "imed_do"), ("imed", "imed_md")):
        base_model = validation_model()
        base_model["paraset"][[0, 12], 1] = nominal[[0, 12]]
        models = [base_model]
        if suffix == "imed_md":
            alternative = {"Name": "Validation PBM with fitted growth order", "paraset": base_model["paraset"].copy()}
            alternative["paraset"][13] = (1, 1., .8, 1.25)
            models.append(alternative)
        case = {"Solubility": solubility, "pbmsetting": {**mesh, "Numerics": "conservative"}, "ModelParameter": models,
                "objweightfactor": {"W1": 0., "W2": 1., "W3": 1., "Terms": "D10 D50 D90"},
                "pyfitsettings": {"global_iterations": 0, "population_size": 5, "local_evaluations": 65}}
        modeling = pack_imed_settings(case)
        if backend == "bo":
            bo = json.loads((examples / "bo_validation_loop.json").read_text(encoding="utf-8"))
            bo["source"] = "Calibrated pump profiles"
            bo["physical_profiles"] = {}
            modeling.update(active_workflow="bo", bo=bo)
        loop = {**setup, "imed": {**setup["imed"], "delta_aic": 600}} if suffix == "imed_md" else setup
        environment = {"prepare_simulated_loop": "true", "loop_backend": backend,
                       "loop_setup": json.dumps(loop), "experiment_count": str(len(frames))}
        write_preset(output / f"calibrated_{suffix}_setup_preset.json", panel, ExperimentProfileParameters(), (),
                     environment, {"output_folder": str(output), "sql_enabled": False},
                     monitor_device_ids=tuple(MONITORS.values()), modeling=modeling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("calibration_config", type=Path)
    args = parser.parse_args()
    generate(args.output, args.calibration_config)
