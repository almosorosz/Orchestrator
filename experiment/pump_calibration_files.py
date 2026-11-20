"""Portable per-device calibration snapshots loaded from CSV files."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.pump_curve_fits import fit_seed_exponential, monotone_quadratic, seed_exponential_mass, seed_exponential_time

OUTPUT_UNITS = {"masterflex": "RPM", "ismatec": "RPM"}


def validate_calibration(calibration, device_type):
    if calibration.get("version") != 1 or device_type not in OUTPUT_UNITS:
        raise ValueError("Unsupported pump calibration")
    unit = calibration.get("output_unit")
    if (calibration.get("device_type", device_type) != device_type or calibration.get("input_unit") != "g/h"
            or (unit != OUTPUT_UNITS[device_type] and not (device_type == "ismatec" and unit == "mL/min"))):
        raise ValueError(f"Calibration must convert g/h to the configured {device_type} command units.")
    points = np.asarray(calibration.get("points"), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Calibration needs at least two finite flow/target pairs")
    if not np.all(points[0] == 0) or np.any(points < 0) or np.any(np.diff(points[:, 0]) <= 0) or np.any(np.diff(points[:, 1]) <= 0):
        raise ValueError("Calibration must start at (0, 0) and flow and pump targets must increase")
    if "flow_fit" in calibration:
        coefficients = np.asarray(calibration["flow_fit"], dtype=float)
        if coefficients.shape != (2,) or not np.isfinite(coefficients).all():
            raise ValueError("The liquid calibration needs finite rate coefficients.")
        b, c = coefficients
        limit = points[-1, 1]
        if b < 0 or b + 2*c*limit < -1e-12*max(b, 1.) or b + c*limit <= 0:
            raise ValueError("The fitted delivery rate must increase within its calibrated setting range.")
    return points


def load_calibration(filename, device_type, flow_column, target_column, *, header_row=0,
                     mode="pairs", time_column=None, time_unit="min", mass_column=None, flow_unit="g/h"):
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
    if "ml/min" in declared or "ml_min" in declared or first_value in {"ml/min", "ml/minute"}:
        raise ValueError("Device commands use RPM. Open Define pump profile to review a log labeled mL/min.")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    # Units/header rows are allowed; partially numeric data rows are malformed.
    numeric = numeric.dropna(how="all")
    if numeric.empty or numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Calibration contains missing/non-numeric measurements")
    plateau_rates = None
    if mode == "pairs":
        points = numeric[[flow_column, target_column]].to_numpy(dtype=float)
        points[:, 0] *= {"g/h": 1., "g/min": 60., "g/s": 3600.}[flow_unit]
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
        plateau_rates = [list(pair) for pair in pairs]
        points = np.asarray(pairs)
        # Repeated target plateaus are combined into a single measured rate.
        points = np.asarray([(float(np.mean(points[points[:, 1] == command, 0])), command)
                             for command in np.unique(points[:, 1])])
    points = points[np.argsort(points[:, 0])]
    origin_added = bool(len(points) and points[0, 0] > 0)
    if origin_added:
        points = np.vstack(([0., 0.], points))
    result = {"version": 1, "device_type": device_type, "source_name": filename.name, "source_file": str(filename.resolve()),
              "sha256": sha256(filename.read_bytes()).hexdigest(), "format": mode,
              "input_unit": "g/h", "output_unit": OUTPUT_UNITS[device_type], "points": points.tolist(),
              "origin_added": origin_added,
              "columns": {"flow": flow_column, "target": target_column, "time": time_column, "mass": mass_column},
              "header_row": header_row, "time_unit": time_unit if mode == "mass_log" else None}
    if plateau_rates is not None:
        result["plateau_rates"] = plateau_rates
    validate_calibration(result, device_type)
    return result


def fit_liquid_flow(calibration, device_type, density=None):
    points = validate_calibration(calibration, device_type)
    calibration["flow_fit"] = monotone_quadratic(points[:, 1], points[:, 0])
    if density is not None:
        if not np.isfinite(density) or density <= 0:
            raise ValueError("Enter a positive liquid density to show mL/min.")
        calibration["density_g_ml"] = float(density)
    validate_calibration(calibration, device_type)
    return calibration


def convert_flow(calibration, device_type, values):
    points = validate_calibration(calibration, device_type)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Flow targets must be finite")
    magnitude = np.abs(values)
    fit = calibration.get("flow_fit")
    rpm = points[-1, 1]
    limit = rpm * (fit[0] + fit[1]*rpm) if fit is not None else points[-1, 0]
    if np.any(magnitude > limit + 1e-9):
        raise ValueError(f"Requested flow exceeds calibration range 0–{limit:g} g/h. Load a calibration covering the requested range.")
    if fit is None:
        return np.sign(values) * np.interp(magnitude, points[:, 0], points[:, 1])
    b, c = fit
    # Rationalized quadratic root on the increasing branch, including c = 0.
    denominator = b + np.sqrt(np.maximum(0., b*b + 4*c*np.minimum(magnitude, limit)))
    commands = np.divide(2*magnitude, denominator, out=np.zeros_like(magnitude), where=denominator > 0)
    return np.sign(values) * np.minimum(commands, rpm)


def delivered_flow(config, targets):
    """Invert the loaded calibration: device commands → delivered mass in g/h."""
    calibration = config.get("pump_calibration", {})
    points = validate_calibration(calibration, config["type"])
    targets = np.asarray(targets, dtype=float)
    if not np.isfinite(targets).all() or np.any(np.abs(targets) > points[-1, 1] + 1e-9):
        raise ValueError("Pump commands exceed the loaded calibration range; extrapolation is not supported.")
    if "flow_fit" in calibration:
        b, c = calibration["flow_fit"]
        return targets * (b + c*np.abs(targets))
    return np.sign(targets) * np.interp(np.abs(targets), points[:, 1], points[:, 0])


def calibration_knots(config, times, values, *, inverse=False):
    """Split a linear ramp at every slope change of the measured pump curve."""
    calibration = config["pump_calibration"]
    points = validate_calibration(calibration, config["type"])
    if "flow_fit" in calibration:
        b, c = calibration["flow_fit"]
        # The equipment runner interpolates commands linearly between knots.
        speeds = np.linspace(0., points[-1, 1], 65) if c else points[[0, -1], 1]
        knots = speeds if inverse else speeds * (b + c*speeds)
    else:
        knots = points[:, int(inverse)]
    knots = np.r_[-knots[:0:-1], knots]
    extra = []
    for start, end, a, b in zip(times, times[1:], values, values[1:]):
        between = knots[(knots > min(a, b)) & (knots < max(a, b))]
        if between.size:
            crossing = start + (end - start) * (between - a) / (b - a)
            extra.extend(crossing[(crossing > start) & (crossing < end)])
    if not extra:
        return times, values
    result = np.unique(np.r_[times, extra])
    return result, np.interp(result, times, values)


def seed_delivery_rate(config, settings):
    if settings.get("use_device_calibration", False):
        return float(delivered_flow(config, [settings["rpm"]])[0]) / 3600.
    return settings["grams_per_second"]


def validate_seed_calibration(calibration, config):
    from hardware_devices.limits import target_allowed, target_limits

    rpm = calibration.get("rpm")
    if (calibration.get("version") != 1 or config["type"] != "masterflex" or type(rpm) not in (int, float)
            or rpm <= 0 or not target_allowed(rpm, target_limits(config))):
        raise ValueError("The seed calibration needs a fixed RPM within the assigned Masterflex pump's limits.")
    points = np.asarray(calibration.get("points"), dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(np.diff(points[:, 0]) <= 0) or not np.all(points[0] == 0)
            or np.any(np.diff(points[:, 1]) < 0) or np.any(np.diff(points[:, 1][points[:, 1] > 0]) <= 0)
            or points[-1, 1] <= 0):
        raise ValueError("Seed calibration needs increasing pumping times and strictly increasing positive doses in dry seed grams.")
    if "seed_fit" in calibration:
        model = calibration["seed_fit"]
        delay, rate, k = (model.get(name, np.nan) for name in ("delay_s", "initial_rate_g_s", "k_s_inv"))
        if not np.isfinite([delay, rate, k]).all() or delay != points[points[:, 1] == 0, 0][-1] or rate <= 0 or k < 0:
            raise ValueError("Invalid exponential seed calibration.")
    return points


def load_seed_calibration(filename, config, time_column, mass_column, rpm, *, header_row=0, time_unit="s",
                          target_column=None, reset_clock=False, fit_exponential=False):
    filename = Path(filename)
    if time_column == mass_column:
        raise ValueError("Select distinct time and dry seed mass columns.")
    frame = pd.read_csv(filename, header=header_row)
    columns = [time_column, mass_column] + ([target_column] if target_column else [])
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if numeric.empty or numeric.isna().any().any():
        raise ValueError("Seed calibration contains missing or non-numeric measurements.")
    if target_column:
        numeric = numeric.loc[np.isclose(numeric[target_column], rpm)]
        if numeric.empty:
            raise ValueError(f"The file has no dry-seed measurements at {rpm:g} RPM.")
    points = numeric[[time_column, mass_column]].to_numpy(dtype=float, copy=True)
    points[:, 0] *= {"s": 1., "min": 60., "h": 3600.}[time_unit]
    points = points[np.argsort(points[:, 0])]
    if reset_clock:
        points[:, 0] -= points[0, 0]
    origin_added = bool(len(points) and points[0, 0] > 0)
    if origin_added:
        points = np.vstack(([0., 0.], points))
    result = {"version": 1, "source_name": filename.name, "source_file": str(filename.resolve()),
              "sha256": sha256(filename.read_bytes()).hexdigest(), "rpm": rpm, "points": points.tolist(),
              "origin_added": origin_added,
              "columns": {"time": time_column, "mass": mass_column, "target": target_column},
              "header_row": header_row, "time_unit": time_unit, "reset_clock": reset_clock}
    validate_seed_calibration(result, config)
    if fit_exponential:
        result["seed_fit"] = fit_seed_exponential(points)
    return result


def seed_dose_speed(config, settings=None):
    calibration = config.get("seed_calibration")
    if calibration and (settings is None or settings.get("use_device_calibration")):
        validate_seed_calibration(calibration, config)
        return calibration["rpm"]
    return settings["rpm"] if settings else None


def seed_dose_command(config, settings, grams):
    rpm = seed_dose_speed(config, settings)
    calibration = config.get("seed_calibration")
    if calibration and (settings is None or settings.get("use_device_calibration")):
        points = validate_seed_calibration(calibration, config)
        if not np.isfinite(grams) or grams < 0:
            raise ValueError("Enter a finite, nonnegative dry seed dose.")
        if grams == 0:
            return rpm, 0.
        if "seed_fit" in calibration:
            return rpm, seed_exponential_time(calibration["seed_fit"], grams)
        if grams > points[-1, 1] + 1e-9:
            raise ValueError(f"Requested seed mass {grams:g} g exceeds the calibrated range 0–{points[-1, 1]:g} g.")
        # A zero-mass plateau may describe the delay before the first solids emerge.
        last_zero = np.searchsorted(points[:, 1], 0., side="right") - 1
        positive = points[last_zero:]
        return rpm, float(np.interp(grams, positive[:, 1], positive[:, 0]))
    if not settings:
        raise ValueError("Configure calibrated seed delivery for the assigned seed pump.")
    return rpm, grams / seed_delivery_rate(config, settings)


def seed_mass_from_profile(profile, config, end_s):
    """Infer dry seed from one fixed-speed pulse during preparation."""
    calibration = config["seed_calibration"]
    points = validate_seed_calibration(calibration, config)
    rpm = calibration["rpm"]
    t = np.array([p.time_s for p in profile.points])
    y = np.array([p.target for p in profile.points])
    bounds = np.unique(np.r_[0., t[(t > 0) & (t < end_s)], end_s])
    left, right = bounds[:-1], bounds[1:]
    width = right - left
    if profile.interpolation == "step":
        targets = y[np.clip(np.searchsorted(t, left, side="right") - 1, 0, len(y) - 1)]
        active = targets > 0
        fixed = np.isclose(targets, rpm)
    else:
        starts, stops = np.interp(left, t, y), np.interp(right, t, y)
        active = (np.maximum(starts, stops) > 0) & (width > 1e-5)
        fixed = np.isclose(starts, rpm) & np.isclose(stops, rpm)
    if np.any(active & ~fixed):
        raise ValueError(f"Seed pump {profile.device_id} must use a fixed {rpm:g} RPM pulse to infer seed grams. Enter a seed mass override for other profiles.")
    pulses = np.flatnonzero(active)
    if not pulses.size:
        return 0.
    if width[pulses[0]:pulses[-1]+1][~active[pulses[0]:pulses[-1]+1]].sum() > 1e-5:
        raise ValueError(f"Seed pump {profile.device_id} has multiple pulses. Enter the dry seed mass as an experiment override.")
    seconds = width[active].sum()
    if "seed_fit" in calibration:
        return float(seed_exponential_mass(calibration["seed_fit"], seconds))
    if seconds > points[-1, 0] + 1e-5:
        raise ValueError(f"Seed pulse for {profile.device_id} is {seconds:g} s, beyond the calibrated range (up to {points[-1, 0]:g} s).")
    return float(np.interp(seconds, points[:, 0], points[:, 1]))
