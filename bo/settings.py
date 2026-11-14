"""Versioned BO setup documents with embedded data for portable presets."""
from copy import deepcopy
import math

from .profiles import CHANNELS, CONSTRAINTS, TARGETS, pairs

FORMAT = "crystalloop-bo"
OUTPUTS = ("yield", "d10", "d50", "d90")
CHECKPOINT_CHANNELS = ("temperature", "feed")


def empty_settings():
    return {
        "format": FORMAT, "version": 1, "source": "", "runs": [], "selected": [],
        "constraints": {key: [None, None] for key in CONSTRAINTS},
        "targets": {key: {"enabled": key == "duration", "mode": modes[0], "weight": 1.,
                          "target": 0., "scale": 1.}
                    for key, (_, _, modes) in TARGETS.items()},
        "yield_basis": "theoretical", "overrides": {},
        "output_bounds": {key: [None, None] for key in OUTPUTS},
        "exploration": 1., "output_margin": 1.96, "vae_checkpoints": {},
        "tolerances": {key: entry[2] for key, entry in CHANNELS.items()},
    }


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_settings(value):
    if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("version") != 1:
        raise ValueError("Choose a CrystalLoop BO setup file.")
    # Existing component files and presets predate these optional fields.
    data = deepcopy(value)
    defaults = empty_settings()
    for key in ("output_bounds", "exploration", "output_margin", "vae_checkpoints"):
        data.setdefault(key, defaults[key])
    try:
        if data["yield_basis"] not in ("theoretical", "actual") or not isinstance(data["source"], str):
            raise ValueError("Unknown yield basis or source.")
        for field, definitions in (("constraints", CONSTRAINTS), ("output_bounds", OUTPUTS)):
            if set(data[field]) != set(definitions):
                raise ValueError("Unknown constraint fields.")
            for key in definitions:
                lower, upper = data[field][key]
                label = CONSTRAINTS[key][0] if field == "constraints" else TARGETS[key][0]
                if any(v is not None and (not finite(v) or (key != "yield" and v < 0))
                       for v in (lower, upper)):
                    raise ValueError(f"Invalid minimum or maximum for {label}.")
                if lower is not None and upper is not None and lower > upper:
                    raise ValueError(f"{label} minimum exceeds its maximum.")
        if any(not finite(data[key]) or data[key] < 0 for key in ("exploration", "output_margin")):
            raise ValueError("Exploration and the output margin must be finite and nonnegative.")
        for key, checkpoint in data["vae_checkpoints"].items():
            if key not in CHECKPOINT_CHANNELS or not isinstance(checkpoint["name"], str):
                raise ValueError("Invalid trajectory VAE checkpoint.")
            if not isinstance(checkpoint["model_state_dict"], dict):
                raise ValueError("A trajectory VAE checkpoint is missing its weights.")
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
