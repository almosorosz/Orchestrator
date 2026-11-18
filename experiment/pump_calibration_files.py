"""Portable per-device calibration snapshots loaded from CSV files."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_UNITS = {"masterflex": "RPM", "ismatec": "mL/min"}


def validate_calibration(calibration, device_type):
    if calibration.get("version") != 1 or device_type not in OUTPUT_UNITS:
        raise ValueError("Unsupported pump calibration")
    if calibration.get("input_unit") != "g/h" or calibration.get("output_unit") != OUTPUT_UNITS[device_type]:
        raise ValueError(f"Calibration must convert g/h to {OUTPUT_UNITS[device_type]}")
    points = np.asarray(calibration.get("points"), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Calibration needs at least two finite flow/target pairs")
    if not np.all(points[0] == 0) or np.any(points < 0) or np.any(np.diff(points[:, 0]) <= 0) or np.any(np.diff(points[:, 1]) <= 0):
        raise ValueError("Calibration must start at (0, 0) and flow and pump targets must increase")
    return points


def load_calibration(filename, device_type, flow_column, target_column, *, header_row=0,
                     mode="pairs", time_column=None, time_unit="min", mass_column=None):
    filename = Path(filename)
    frame = pd.read_csv(filename, header=header_row)
    if device_type not in OUTPUT_UNITS:
        raise ValueError("Select a Masterflex or Ismatec pump")
    required = [flow_column, target_column] if mode == "pairs" else [time_column, target_column, mass_column]
    if mode not in {"pairs", "mass_log"} or any(column not in frame for column in required):
        raise ValueError("Select valid calibration columns")
    if len(set(required)) != len(required):
        raise ValueError("Choose distinct columns for each calibration measurement")
    declared = target_column.lower()
    first_value = str(frame[target_column].iloc[0]).strip().lower() if len(frame) else ""
    if device_type == "ismatec" and ("rpm" in declared or first_value == "rpm"):
        raise ValueError("Ismatec calibration targets must be mL/min, not RPM")
    if device_type == "masterflex" and ("ml/min" in declared or "ml_min" in declared or first_value in {"ml/min", "ml/minute"}):
        raise ValueError("Masterflex calibration targets must be RPM, not mL/min")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    # Units/header rows are allowed; partially numeric data rows are malformed.
    numeric = numeric.dropna(how="all")
    if numeric.empty or numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Calibration contains missing/non-numeric measurements")
    if mode == "pairs":
        points = numeric[[flow_column, target_column]].to_numpy(dtype=float)
    else:
        scale = {"s": 1 / 3600, "min": 1 / 60, "h": 1}.get(time_unit)
        if scale is None:
            raise ValueError("Choose seconds, minutes or hours for the log timeline")
        times = numeric[time_column].to_numpy(dtype=float) * scale
        commands = numeric[target_column].to_numpy(dtype=float)
        masses = numeric[mass_column].to_numpy(dtype=float)
        if np.any(np.diff(times) <= 0):
            raise ValueError("Calibration log time must increase strictly")
        boundaries = [0, *(np.flatnonzero(~np.isclose(commands[1:], commands[:-1])) + 1), len(commands)]
        pairs = []
        for start, stop in zip(boundaries, boundaries[1:]):
            if stop - start < 2 or commands[start] <= 0:
                continue
            rate = abs(float(np.polyfit(times[start:stop] - times[start], masses[start:stop], 1)[0]))
            if rate > 0:
                pairs.append((rate, float(commands[start])))
        if not pairs:
            raise ValueError("Log needs positive constant-target segments with at least two mass/time readings each")
        points = np.asarray(pairs)
        # Repeated target plateaus are combined into a single measured rate.
        points = np.asarray([(float(np.mean(points[points[:, 1] == command, 0])), command)
                             for command in np.unique(points[:, 1])])
    points = points[np.argsort(points[:, 0])]
    if len(points) and points[0, 0] > 0:
        points = np.vstack(([0., 0.], points))
    result = {"version": 1, "source_name": filename.name, "source_file": str(filename.resolve()),
              "sha256": sha256(filename.read_bytes()).hexdigest(), "format": mode,
              "input_unit": "g/h", "output_unit": OUTPUT_UNITS[device_type], "points": points.tolist(),
              "columns": {"flow": flow_column, "target": target_column, "time": time_column, "mass": mass_column},
              "header_row": header_row, "time_unit": time_unit if mode == "mass_log" else None}
    validate_calibration(result, device_type)
    return result


def convert_flow(calibration, device_type, values):
    points = validate_calibration(calibration, device_type)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Flow targets must be finite")
    if np.any(np.abs(values) > points[-1, 0] + 1e-9):
        raise ValueError(f"Requested flow exceeds calibration range 0–{points[-1, 0]:g} g/h. Load a calibration covering the requested range.")
    return np.sign(values) * np.interp(np.abs(values), points[:, 0], points[:, 1])
