"""Generate a loadable DoE preset for the validation PBM demonstration."""
import argparse
import json
from pathlib import Path

import pandas as pd

from app.presets import write_preset
from experiment.doe_campaign import ProfileAssignment
from experiment.profiles.profile_gen import ExperimentProfileParameters
from imed_python.demo_campaign import VERSION


ASSIGNMENTS = {"T": "T1", "AS": "PAS1", "seed": "PSEED1", "clean_sol": "PCLEN1", "valve_state": "V1"}
MONITORS = {"Concentration": "IR1", "FBRM": "FBRM1", "D10": "D101", "D50": "D501",
            "D90": "D901", "AS scale": "WAS1"}


def generate(path, recording_folder):
    root = Path(__file__).resolve().parents[1]
    example = json.loads((root / "examples/configurations/crystallization_system_simulated.json")
                         .read_text(encoding="utf-8"))
    devices = {key: example["devices"][key] for key in (*ASSIGNMENTS.values(), *MONITORS.values())}
    devices["PCLEN1"].update(type="ismatec", connection={"port": ""}, defaults={"stage": "2"})
    for device in MONITORS.values():
        devices[device]["name"] = devices[device]["name"].replace(" (simulated placeholder)", "")
    for config in devices.values():
        config["simulation"]["noise"] = 0
    times = [0, 2, 10, 25, 35, 60, 61, 68, 69, 70, 74, 75, 78, 83]
    frames, specs, conditions = {}, {}, {}
    cases = (("Seeded cooling", True, True, False),
             ("Unseeded cooling and antisolvent", False, True, True),
             ("Seeded isothermal antisolvent", True, False, True))
    for index, (label, seeded, cooling, adding) in enumerate(cases, 1):
        key = f"exp{index}"
        frame = pd.DataFrame({
            "t": [time / 3600 for time in times],
            "T": [25, 25, 22, 16] + [12] * 10 if cooling else [25] * len(times),
            "AS": [0, 0, 1440, 1440] + [0] * 10 if adding else [0] * len(times),
            "seed": [60 if seeded else 0] + [0] * 13,
            "clean_sol": [0] * 9 + [360, 360, 0, 0, 0],
            "valve_state": [0] * 6 + [1, 0, 0, 0, 0, 0, 1, 0],
            "solution": 0, "solvent": 0,
        })
        frames[key] = frame.to_dict(orient="split")
        specs[key] = {"imed_simulation": {"version": VERSION, "label": label, "seeded": seeded,
                                          "start_s": 0, "hold_start_s": 35, "end_s": 60,
                                          "monitors": MONITORS}}
        conditions[key] = {"seeded": int(seeded), "final_T": 12 if cooling else 25,
                           "antisolvent": int(adding)}

    names = ("seeded", "final_T", "antisolvent")
    samples = [[row[name] for name in names] for row in conditions.values()]
    snapshot = {
        "version": 1,
        "storage": {"data": frames, "data_omni": frames, "profile": {}, "profile_omni": {},
                    "DoE": {key: pd.DataFrame([row]).to_dict(orient="split")
                            for key, row in conditions.items()}},
        "attributes": {"TYPE": "IMED validation demo", "dimensions": 3, "samples": 3,
                       "Fract": 1, "if_sampling": False, "actually_sampled_experiments": 3},
        "arrays": {"DoE_samples": samples, "DoE_sampled_scaled": [[1, 0, 0], [0, 0, 1], [1, 1, 1]]},
        "DoE_val": {name: [row[name] for row in conditions.values()] for name in names},
        "weights": [1, 1, 1], "parameter_snapshots": specs,
    }
    write_preset(path, {"devices": devices, "profiles": {}, "experiments": [],
                        "recording_folder": str(recording_folder), "simulation_speed": 100,
                        "system_topology": example["system_topology"]},
                 ExperimentProfileParameters(),
                 tuple(ProfileAssignment(role, key) for role, key in ASSIGNMENTS.items()),
                 {"experiment_count": "3"}, {"output_folder": str(recording_folder), "sql_enabled": False},
                 generated_doe=snapshot, monitor_device_ids=tuple(MONITORS.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", type=Path)
    parser.add_argument("--recordings", type=Path, default=Path("data/imed_simulated_runs"))
    args = parser.parse_args()
    generate(args.preset, args.recordings)
