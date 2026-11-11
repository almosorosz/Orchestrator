"""Versioned, non-executable setup presets; no connections or execution state."""
from dataclasses import asdict, fields
import json
import math
import pandas as pd
from pathlib import Path

from experiment.profiles.profile_gen import ExperimentProfileParameters, DefineProcessParameter
from experiment.doe_campaign import ProfileAssignment, ROLES
from experiment.models import ExperimentDefinition
from experiment.recording import validate_settings


def read_preset(filename):
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    if data.get("format") != "crystalloop-setup" or data.get("version") not in (1, 2):
        raise ValueError("Unsupported setup preset format/version")
    panel = data["control_panel"]
    from control_panel.control_panel import HARDWARE_TYPES
    devices = panel["devices"]
    from experiment.system_topology import validate_topology, empty_topology
    validate_topology(panel.get("system_topology", empty_topology()), devices, complete=False)
    if not isinstance(devices, dict) or not devices:
        raise ValueError("Preset needs at least one configured device")
    for key, config in devices.items():
        if not isinstance(key, str) or not key or config.get("type") not in HARDWARE_TYPES:
            raise ValueError(f"Invalid device: {key}")
        if not isinstance(config.get("connection", {}), dict):
            raise ValueError(f"Invalid connection settings for {key}")
        if config.get("pump_calibration"):
            from experiment.pump_calibration_files import validate_calibration
            validate_calibration(config["pump_calibration"], config["type"])
    for frame in panel.get("experiments", []):
        ExperimentDefinition.from_frame(pd.DataFrame(**frame), devices)
    for key, saved in panel.get("profiles", {}).items():
        frame = pd.DataFrame(**saved)
        if key not in devices or not {"time_s", "value"}.issubset(frame.columns):
            raise ValueError(f"Invalid editable profile for {key}")
        ExperimentDefinition.from_frame(frame[["time_s", "value"]].rename(columns={"value": key}), devices)
    environment = data.get("environment", {})
    if not isinstance(environment, dict) or any(not isinstance(value, str) for value in environment.values()):
        raise ValueError("Environment paths and sampling settings must be text")
    from imed_python.size_data import FINAL_SIZE_TERM, SIZE_SAMPLE_OPTIONS
    if environment.get("pbm_size_term", FINAL_SIZE_TERM) not in {value for _, value in SIZE_SAMPLE_OPTIONS}:
        raise ValueError("Unknown LOOP PBM size-sample setting")
    if not isinstance(panel.get("recording_folder", ""), str):
        raise ValueError("Invalid Control Panel recording folder")
    speed = panel.get("simulation_speed", 1)
    if not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
        raise ValueError("Invalid simulation speed")
    parameters = ExperimentProfileParameters()
    provided = data.get("process", {})
    if set(provided) - {field.name for field in fields(parameters)}:
        raise ValueError("Unknown process parameter in preset")
    for field in fields(parameters):
        if field.name not in provided:
            continue
        value = provided[field.name]
        default = getattr(parameters, field.name)
        if isinstance(default, DefineProcessParameter):
            value = DefineProcessParameter(**value)
            if not all(math.isfinite(float(v)) for v in (value.value, value.min_val, value.max_val)) or (value.if_in_doe and value.min_val > value.max_val):
                raise ValueError(f"Invalid bounds for {field.name}")
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{field.name} must be true or false")
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError(f"Invalid text for {field.name}")
        elif isinstance(default, list):
            if not isinstance(value, list) or len(value) != len(default):
                raise ValueError(f"Invalid list for {field.name}")
            for entry, original in zip(value, default):
                if isinstance(original, str):
                    if not isinstance(entry, str):
                        raise ValueError(f"Invalid list value for {field.name}")
                elif not isinstance(entry, (int, float)) or not math.isfinite(entry):
                    raise ValueError(f"Invalid numeric list for {field.name}")
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid value for {field.name}")
        setattr(parameters, field.name, value)
    assignments = tuple(ProfileAssignment(**row) for row in data.get("assignments", []))
    if len({a.role for a in assignments}) != len(assignments):
        raise ValueError("Duplicate profile roles in preset")
    for assignment in assignments:
        if assignment.role not in ROLES or assignment.device_id not in devices or devices[assignment.device_id]["type"] not in ROLES[assignment.role].compatible_types:
            raise ValueError(f"Invalid assignment: {assignment.role}")
    monitors = data.get("monitor_device_ids", [])
    if not isinstance(monitors, list) or any(not isinstance(key, str) for key in monitors) or len(set(monitors)) != len(monitors):
        raise ValueError("Invalid saved monitor IDs")
    data["recording"] = validate_settings(data["recording"])
    from experiment.doe_snapshot import validate_snapshot
    validate_snapshot(data.get("generated_doe"))
    return data, parameters, assignments


def write_preset(filename, panel, parameters, assignments, environment, recording, generated_doe=None, *, monitor_device_ids=()):
    from app.version import __version__
    payload = {"format": "crystalloop-setup", "version": 2, "application_version": __version__,
               "control_panel": panel, "process": asdict(parameters),
               "assignments": [asdict(a) for a in assignments], "environment": environment,
               "recording": validate_settings(recording), "generated_doe": generated_doe,
               "monitor_device_ids": list(monitor_device_ids)}
    from experiment.recording import atomic_json
    atomic_json(Path(filename), payload)
