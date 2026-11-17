"""Generate a loadable DoE preset for the validation PBM demonstration."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.presets import write_preset
from experiment.doe_campaign import ProfileAssignment
from experiment.profiles.profile_gen import ExperimentProfileParameters
from imed_python.demo_campaign import VERSION, model_inputs, validation_model
from imed_python.setup_snapshot import pack_imed_settings


ASSIGNMENTS = {"T": "T1", "AS": "PAS1", "seed": "PSEED1", "clean_sol": "PCLEN1", "valve_state": "V1"}
MONITORS = {"Concentration": "IR1", "FBRM": "FBRM1", "D10": "D101", "D50": "D501",
            "D90": "D901", "AS scale": "WAS1"}
IMED_CASES = (("Seeded cooling to 12 °C", True, 12, 0),
              ("Unseeded cooling to 12 °C with antisolvent", False, 12, 1440),
              ("Seeded isothermal at 25 °C with antisolvent", True, 25, 1440),
              ("Seeded cooling to 16 °C with antisolvent", True, 16, 1080),
              ("Unseeded cooling to 19 °C with antisolvent", False, 19, 1620))
BO_CASES = tuple((f"Seeded isothermal at {temp:g} °C", True, temp, 0)
                 for temp in (13, 15.5, 18, 20.5, 23))


def generate(path, recording_folder, *, mode=None):
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples" / "configurations"
    example = json.loads((examples / "crystallization_system_simulated.json")
                         .read_text(encoding="utf-8"))
    devices = {key: example["devices"][key] for key in (*ASSIGNMENTS.values(), *MONITORS.values())}
    devices["PCLEN1"].update(type="ismatec", connection={"port": ""}, defaults={"stage": "2"})
    for device in MONITORS.values():
        devices[device]["name"] = devices[device]["name"].replace(" (simulated placeholder)", "")
    for config in devices.values():
        config["simulation"]["noise"] = 0
    times = [0, 2, 10, 25, 35, 60, 61, 68, 69, 70, 74, 75, 78, 83]
    frames, specs, conditions = {}, {}, {}
    cases = BO_CASES if mode == "bo" else IMED_CASES if mode == "imed" else (
        ("Seeded cooling", True, 12, 0), ("Unseeded cooling and antisolvent", False, 12, 1440),
        ("Seeded isothermal antisolvent", True, 25, 1440),
    )
    feed_factor = "antisolvent_g_h" if mode else "antisolvent"
    for index, (label, seeded, final_temp, feed) in enumerate(cases, 1):
        key = f"exp{index}"
        if mode == "bo":
            temperature = [final_temp] * len(times)
        else:
            temperature = [25, 25, min(25, final_temp + 10), min(25, final_temp + 4)] + [final_temp] * 10
        frame = pd.DataFrame({
            "t": [time / 3600 for time in times],
            "T": temperature,
            "AS": [0, 0, feed, feed] + [0] * 10,
            "seed": [60 if seeded else 0] + [0] * 13,
            "clean_sol": [0] * 9 + [360, 360, 0, 0, 0],
            "valve_state": [0] * 6 + [1, 0, 0, 0, 0, 0, 1, 0],
            "solution": 0, "solvent": 0,
        })
        frames[key] = frame.to_dict(orient="split")
        specs[key] = {"imed_simulation": {"version": VERSION, "label": label, "seeded": seeded,
                                          "start_s": 0, "hold_start_s": 35, "end_s": 60,
                                          "monitors": MONITORS}}
        conditions[key] = {"seeded": int(seeded), "final_T": final_temp,
                           feed_factor: feed if mode else int(bool(feed))}

    names = ("seeded", "final_T", feed_factor)
    samples = np.asarray([[row[name] for name in names] for row in conditions.values()])
    span = np.ptp(samples, axis=0)
    scaled = (samples - samples.min(axis=0)) / np.where(span, span, 1)
    count = len(cases)
    snapshot = {
        "version": 1,
        "storage": {"data": frames, "data_omni": frames, "profile": {}, "profile_omni": {},
                    "DoE": {key: pd.DataFrame([row]).to_dict(orient="split")
                            for key, row in conditions.items()}},
        "attributes": {"TYPE": "IMED validation demo" if mode is None else f"{mode.upper()} simulated loop",
                       "dimensions": 3, "samples": count, "Fract": 1, "if_sampling": False,
                       "actually_sampled_experiments": count},
        "arrays": {"DoE_samples": samples.tolist(), "DoE_sampled_scaled": scaled.tolist()},
        "DoE_val": {name: [row[name] for row in conditions.values()] for name in names},
        "weights": [1, 1, 1], "parameter_snapshots": specs,
    }
    environment = {"experiment_count": str(count)}
    modeling = None
    if mode:
        _, _, _, solubility, mesh = model_inputs(True)
        case = {"Solubility": solubility, "pbmsetting": {**mesh, "Numerics": "conservative"}}
        setup = {"max_followups": 1}
        if mode == "imed":
            case.update(ModelParameter=[validation_model()],
                        objweightfactor={"W1": 0., "W2": 1., "W3": 1., "Terms": "D10 D50 D90"},
                        pyfitsettings={"global_iterations": 0, "population_size": 5, "local_evaluations": 60})
            setup = json.loads((examples / "imed_validation_loop.json").read_text(encoding="utf-8"))
            setup["imed"]["conditions"].update(temperature=[21, 18], temperature_rates=[0.11, 0.3])
        modeling = pack_imed_settings(case)
        if mode == "bo":
            modeling.update(active_workflow="bo", bo=json.loads(
                (examples / "bo_validation_loop.json").read_text(encoding="utf-8")))
        environment.update(prepare_simulated_loop="true", loop_backend=mode, loop_setup=json.dumps(setup))
    write_preset(path, {"devices": devices, "profiles": {}, "experiments": [],
                        "recording_folder": str(recording_folder), "simulation_speed": 100,
                        "system_topology": example["system_topology"]},
                 ExperimentProfileParameters(),
                 tuple(ProfileAssignment(role, key) for role, key in ASSIGNMENTS.items()),
                 environment, {"output_folder": str(recording_folder), "sql_enabled": False},
                 generated_doe=snapshot, monitor_device_ids=tuple(MONITORS.values()), modeling=modeling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", type=Path)
    parser.add_argument("--recordings", type=Path, default=Path("data/imed_simulated_runs"))
    parser.add_argument("--mode", choices=("imed", "bo"), help="Generate a five-run automatic loop preset.")
    args = parser.parse_args()
    generate(args.preset, args.recordings, mode=args.mode)
