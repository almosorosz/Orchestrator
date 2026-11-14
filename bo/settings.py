"""Versioned BO setup documents with embedded data for portable presets."""
from copy import deepcopy
import math

from .profiles import CHANNELS, CONSTRAINTS, TARGETS, pairs

FORMAT = "crystalloop-bo"


def empty_settings():
    return {
        "format": FORMAT, "version": 1, "source": "", "runs": [], "selected": [],
        "constraints": {key: [None, None] for key in CONSTRAINTS},
        "targets": {key: {"enabled": key == "duration", "mode": modes[0], "weight": 1.,
                          "target": 0., "scale": 1.}
                    for key, (_, _, modes) in TARGETS.items()},
        "yield_basis": "theoretical", "overrides": {},
        "tolerances": {key: entry[2] for key, entry in CHANNELS.items()},
    }


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_settings(value):
    if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("version") != 1:
        raise ValueError("Choose a CrystalLoop BO setup file.")
    data = deepcopy(value)
    try:
        if data["yield_basis"] not in ("theoretical", "actual") or not isinstance(data["source"], str):
            raise ValueError("Unknown yield basis or source.")
        for key in CONSTRAINTS:
            lower, upper = data["constraints"][key]
            if any(v is not None and (not finite(v) or v < 0) for v in (lower, upper)):
                raise ValueError(f"{CONSTRAINTS[key][0]} bounds must be nonnegative numbers.")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{CONSTRAINTS[key][0]} minimum exceeds its maximum.")
        for key, (_, _, modes) in TARGETS.items():
            target = data["targets"][key]
            if (type(target["enabled"]) is not bool or target["mode"] not in modes
                    or not all(finite(target[k]) for k in ("weight", "target", "scale"))
                    or min(target["weight"], target["target"]) < 0 or target["scale"] <= 0):
                raise ValueError(f"Invalid target, weight, or scale for {TARGETS[key][0]}.")
        for key in CHANNELS:
            tolerance = data["tolerances"][key]
            if not finite(tolerance) or tolerance < 0:
                raise ValueError("Encoding tolerances must be nonnegative.")
        runs = data["runs"]
        selected = data["selected"]
        if not isinstance(runs, list) or not isinstance(selected, list) or any(
                type(index) is not int or not 0 <= index < len(runs) for index in selected):
            raise ValueError("The selected experiments are invalid.")
        if len(set(selected)) != len(selected):
            raise ValueError("The selected experiments contain duplicates.")
        for run in runs:
            if not isinstance(run["name"], str) or any(
                    item is not None and not finite(item) for item in run["metrics"].values()):
                raise ValueError("A saved experiment contains invalid measurements.")
            if any(key not in run["metrics"] for key in ("duration", "yield_actual", "yield_theoretical",
                                                        "d10", "d50", "d90")):
                raise ValueError("A saved experiment is missing a response field.")
            if not finite(run["metrics"]["duration"]) or run["metrics"]["duration"] <= 0:
                raise ValueError("Saved experiments require a positive duration.")
            for channel, profile in run["profiles"].items():
                if (channel not in CHANNELS or pairs(profile["points"]) is None
                        or type(profile["step"]) is not bool or not isinstance(profile["source"], str)):
                    raise ValueError("A saved experiment contains an invalid profile.")
            if "temperature" not in run["profiles"]:
                raise ValueError("A saved experiment is missing its temperature profile.")
        for key, enabled in data["overrides"].items():
            if not isinstance(key, str) or type(enabled) is not bool:
                raise ValueError("Invalid decision-variable overrides.")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("The BO settings file is incomplete.") from error
    return data
