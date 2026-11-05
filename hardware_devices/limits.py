"""Editable application limits, shared by validation, drivers and controls."""
import math

DEFAULT_LIMITS = {
    "masterflex": {"min": -300., "max": 300., "min_nonzero": 0., "unit": "RPM"},
    "ismatec": {"min": -1000., "max": 1000., "min_nonzero": 0., "unit": "mL/min"},
    "huber": {"min": -50., "max": 200., "min_nonzero": 0., "unit": "°C"},
    "jkem": {"min": 0., "max": 800., "min_nonzero": 0., "unit": "RPM"},
    "nidaq_valve": {"min": 0., "max": 1., "min_nonzero": 0., "unit": "binary"},
}


def target_limits(config):
    result = dict(DEFAULT_LIMITS[config["type"]])
    result["zero_is_stop"] = config["type"] in {"masterflex", "ismatec", "jkem", "nidaq_valve"}
    result.update({key: float(value) for key, value in config.get("target_limits", {}).items()
                   if key in {"min", "max", "min_nonzero"}})
    if not all(math.isfinite(result[k]) for k in ("min", "max", "min_nonzero")):
        raise ValueError("Limits must be finite")
    if result["min"] > result["max"] or not 0 <= result["min_nonzero"] <= max(abs(result["min"]), abs(result["max"])):
        raise ValueError("Invalid limit range or minimum nonzero target")
    if config["type"] == "nidaq_valve" and (result["min"], result["max"], result["min_nonzero"]) != (0, 1, 0):
        raise ValueError("Valve targets must remain binary (0 or 1)")
    return result


def target_allowed(value, limits):
    if value == 0 and limits.get("zero_is_stop", False):
        return True
    return math.isfinite(value) and limits["min"] <= value <= limits["max"] and (
        value == 0 or abs(value) >= limits["min_nonzero"])
