"""Versioned CrystalLoop IMED case files and their JSON representation."""
import base64
import json
import math
from pathlib import Path

import numpy as np

from .model_file import CHECKPOINT_FIELDS, _atomic_write
from .parameter_labels import PARAMETERS

CASE_FILTER = "IMED cases (*.json *.mat);;CrystalLoop IMED cases (*.json);;MATLAB cases (*.mat)"
NATIVE_FILTER = "CrystalLoop IMED case (*.imed.json)"
MATLAB_FILTER = "MATLAB case (*.mat)"
FORMAT = "crystalloop-imed"


def _pack(value):
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item) for item in value]
    if isinstance(value, np.ndarray):
        kind = value.dtype.kind
        if kind == "O":
            data = [_pack(item) for item in value.flat]
        elif kind in "biufSU":
            data = base64.b64encode(value.tobytes()).decode("ascii")
        else:
            raise ValueError(f"Unsupported case array type {value.dtype}.")
        return {"$array": {"dtype": value.dtype.str, "shape": list(value.shape), "data": data}}
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.generic):
        return _pack(value.item())
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": str(value)}
    return value


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.keys() == {"$bytes"}:
        return base64.b64decode(value["$bytes"], validate=True)
    if value.keys() == {"$float"}:
        scalar = value["$float"]
        if scalar not in ("nan", "inf", "-inf"):
            raise ValueError("Invalid non-finite number in the IMED case.")
        return float(scalar)
    if value.keys() != {"$array"}:
        return {key: _unpack(item) for key, item in value.items()}
    packed = value["$array"]
    dtype = np.dtype(packed["dtype"])
    shape = packed["shape"]
    if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Invalid array dimensions in the IMED case.")
    count = math.prod(shape)
    data = packed["data"]
    if dtype == np.dtype(object) and isinstance(data, list) and len(data) == count:
        array = np.empty(count, dtype=object)
        for index, item in enumerate(data):
            array[index] = _unpack(item)
    elif dtype.kind in "biufSU" and dtype.itemsize and isinstance(data, str):
        raw = base64.b64decode(data, validate=True)
        if len(raw) != count * dtype.itemsize:
            raise ValueError("Invalid array length in the IMED case.")
        array = np.frombuffer(raw, dtype=dtype).copy()
    else:
        raise ValueError("Invalid array type or data in the IMED case.")
    return array.reshape(shape)


def pack_case(case):
    """Pack current data, model definitions, PE controls, and workflow history."""
    packed = {key: _pack(case[key]) for key in CHECKPOINT_FIELDS if key in case}
    return {"format": FORMAT, "version": 1, "case": packed}


def unpack_case(payload):
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") != 1:
        raise ValueError("Choose a CrystalLoop IMED case file.")
    try:
        fields = payload["case"]
        if not isinstance(fields, dict):
            raise ValueError("Missing case data.")
        case = {key: _unpack(fields[key]) for key in CHECKPOINT_FIELDS if key in fields}
        for key in ("ExpData", "ProcInf", "ModelParameter"):
            records = case.get(key, [])
            if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
                raise ValueError(f"Invalid {key} records.")
        for model in case.get("ModelParameter", []):
            table = np.asarray(model.get("paraset"), dtype=float)
            if table.shape != (len(PARAMETERS), 4) or np.isnan(table).any():
                raise ValueError("A model has an invalid kinetic parameter table.")
            model["paraset"] = table
        for key in ("Solubility", "pbmsetting", "pepbmsettings", "pyfitsettings", "peweights", "PythonWorkflow"):
            if key in case and not isinstance(case[key], dict):
                raise ValueError(f"Invalid {key} settings.")
        workflow = case.get("PythonWorkflow", {})
        workers = int(workflow.get("cpu_workers", 1))
        if workers < 1:
            raise ValueError("Invalid CPU worker count.")
        steps = workflow.get("steps", [])
        if not isinstance(steps, (tuple, list)):
            raise ValueError("Invalid workflow history.")
        for step in steps:
            if not isinstance(step, dict) or not {"action", "model_index", "completed_at", "signature", "result"} <= step.keys():
                raise ValueError("Invalid workflow entry.")
            int(step["model_index"])
    except (AttributeError, KeyError, OverflowError, RecursionError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CrystalLoop IMED case. {error}") from error
    return case


def read_native_case(path):
    return unpack_case(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def save_native_case(path, case):
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_name(f"{path.name}.imed.json")
    _atomic_write(path, json.dumps(pack_case(case), allow_nan=False, ensure_ascii=False).encode("utf-8"))
    return path
