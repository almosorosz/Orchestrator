"""Completed IMED workflow steps stored with a MATLAB case."""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json

import numpy as np

from .uncertainty import ParameterIntervals


WORKFLOW_KEY = "PythonWorkflow"


def display_name(record, fallback):
    name = record.get("Name")
    return name.strip() if isinstance(name, str) and name.strip() else fallback


def history(case):
    workflow = case.setdefault(WORKFLOW_KEY, {"version": 1, "steps": []})
    steps = workflow.get("steps", [])
    if isinstance(steps, dict):
        steps = [steps]
    elif isinstance(steps, np.ndarray):
        steps = steps.reshape(-1).tolist()
    workflow["steps"] = list(steps)
    return workflow["steps"]


def signature(case, model_index, *, include_values=True, include_weights=True):
    model = case["ModelParameter"][model_index]
    values = {key: case.get(key, {}) for key in
              ("ExpData", "ProcInf", "Solubility", "pbmsetting", "objweightfactor")}
    values["paraset"] = model["paraset"]
    if include_values:
        values["para"] = np.asarray(model.get("para", []))
    if not include_weights:
        values.pop("objweightfactor", None)

    def normalize(value):
        if isinstance(value, dict):
            fields = {key: normalize(item) for key, item in value.items()}
            # MATLAB struct arrays add empty fields to records that lack them.
            return {key: item for key, item in fields.items()
                    if not (isinstance(item, (dict, list)) and not item)}
        if isinstance(value, (list, tuple)):
            try:
                array = np.asarray(value)
            except ValueError:
                array = np.asarray(value, dtype=object)
            if array.dtype.kind in "biuf":
                return array.astype(float).squeeze().tolist()
            return [normalize(item) for item in value]
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "biuf":
                return value.astype(float).squeeze().tolist()
            return normalize(value.squeeze().tolist())
        if isinstance(value, np.generic):
            return normalize(value.item())
        if isinstance(value, (bool, int, float)):
            return float(value)
        return value

    content = json.dumps(normalize(values), sort_keys=True, allow_nan=True).encode("utf-8")
    return sha256(content).hexdigest()


def record_step(case, action, result, *, model_index=-1, context=""):
    history(case).append({"action": action, "model_index": model_index,
                          "completed_at": datetime.now(timezone.utc).isoformat(),
                          "signature": context, "result": result})


def latest_step(case, action, *, model_index=-1, context=None):
    for step in reversed(history(case)):
        if (step["action"] == action and int(step["model_index"]) == model_index
                and (context is None or step.get("signature") == context)):
            return step
    return None


def pack_intervals(result):
    return asdict(result) if result is not None else {}


def unpack_intervals(values):
    if not isinstance(values, dict) or "covariance" not in values:
        return None
    return ParameterIntervals(
        np.asarray(values["indices"], dtype=int).reshape(-1),
        np.asarray(values["values"], dtype=float).reshape(-1),
        np.atleast_2d(np.asarray(values["covariance"], dtype=float)),
        np.asarray(values["standard_errors"], dtype=float).reshape(-1),
        np.asarray(values["lower"], dtype=float).reshape(-1),
        np.asarray(values["upper"], dtype=float).reshape(-1),
        float(values["confidence"]), int(values["degrees_of_freedom"]),
        str(values["source"]), str(values.get("convention", "matlab")))


def pack_fit(result):
    values = asdict(result)
    values["intervals"] = pack_intervals(result.intervals)
    return values


def unpack_fit(values):
    from .estimation import FitResult

    return FitResult(
        np.asarray(values["parameters"], dtype=float).reshape(-1),
        np.asarray(values["residuals"], dtype=float).reshape(-1),
        np.asarray(values["jacobian"], dtype=float).reshape(-1, np.size(values["parameters"])),
        np.atleast_2d(np.asarray(values["covariance"], dtype=float)),
        float(values["ssr"]), float(values["aic"]), bool(values["converged"]),
        unpack_intervals(values.get("intervals")), str(values.get("uncertainty_error", "")),
        int(values.get("simulation_count", 0)), float(values.get("simulation_seconds", 0.)),
        float(values.get("wall_seconds", 0.)))
