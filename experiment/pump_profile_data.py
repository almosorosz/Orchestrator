"""Read timed pump measurements and keep their operating settings explicit."""
import csv
from hashlib import sha256
from pathlib import Path
import re

import numpy as np
import pandas as pd

from experiment.pump_calibration_files import delivered_flow, fit_liquid_flow
from experiment.pump_curve_fits import monotone_quadratic
from hardware_devices.limits import target_allowed, target_limits


TIME_SCALES = {"s": 1., "min": 60., "h": 3600.}


def declared_time_unit(column, units_row):
    for value in (units_row, column):
        tokens = set(re.findall(r"[a-z]+", str(value).lower()))
        for unit, aliases in (("s", {"s", "sec", "second", "seconds"}), ("min", {"min", "minute", "minutes"}),
                              ("h", {"h", "hr", "hour", "hours"})):
            if tokens & aliases:
                return unit
    return None


def read_profile_log(filename, *, header_row=None, columns=None):
    """Read a CSV with time, command and mass, including the exported title and units rows."""
    path = Path(filename)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if columns is None:
        for header, fields in enumerate(rows[:20]):
            time = next((i for i, field in enumerate(fields) if any(s in field.lower() for s in ("time", "duration"))), None)
            mass = next((i for i, field in enumerate(fields) if any(s in field.lower() for s in ("mass", "weight", "dry", "dose"))), None)
            if time is not None and mass is not None and time != mass:
                break
        else:
            raise ValueError("The CSV needs labeled time, pump setpoint and mass columns.")
        target = next((i for i, field in enumerate(fields) if i not in (time, mass)
                       and any(s in field.lower() for s in ("setpt", "setpoint", "target", "command", "rpm", "flow", "as", "ml/min"))), None)
        if target is None:
            raise ValueError("The CSV needs a labeled pump command column.")
        columns = [fields[i] for i in (time, target, mass)]
    else:
        header = header_row
        columns = list(columns)
        fields = list(pd.read_csv(path, header=header, nrows=0).columns)
        if len(columns) != 3 or len(set(columns)) != 3 or any(column not in fields for column in columns):
            raise ValueError("Select distinct time, pump setpoint and mass columns.")
        time, target, _ = (fields.index(column) for column in columns)
    units = rows[header + 1] if header + 1 < len(rows) else []
    target_text = fields[target].lower() + " " + (units[target].lower() if len(units) > target else "")
    time_unit = declared_time_unit(fields[time], units[time] if len(units) > time else "") or "s"
    declared = "mL/min" if "ml/min" in target_text or "ml_min" in target_text else "RPM" if "rpm" in target_text else None
    data = pd.read_csv(path, header=header, usecols=columns)[columns].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if data.empty or data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError("The calibration CSV contains missing or non-numeric measurements.")
    return {"path": path, "header": header, "columns": columns, "data": data.to_numpy(),
            "time_unit": time_unit, "declared_unit": declared}


def balance_plateaus(log, time_unit):
    """Fit mass versus elapsed time separately for each positive constant setpoint."""
    time, targets, mass = log["data"].T
    time = time * TIME_SCALES[time_unit]
    if np.any(np.diff(time) <= 0):
        raise ValueError("Balance log times must increase strictly.")
    boundaries = np.r_[0, np.flatnonzero(~np.isclose(targets[1:], targets[:-1])) + 1, len(time)]
    plateaus = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if targets[start] <= 0 or stop - start < 2:
            continue
        elapsed = time[start:stop] - time[start]
        weights = mass[start:stop]
        slope = float(np.polyfit(elapsed, weights, 1)[0])
        if abs(slope) <= 1e-12:
            continue
        sample = np.unique(np.linspace(0, len(elapsed) - 1, min(200, len(elapsed)), dtype=int))
        added = np.sign(slope) * (weights - weights[0])
        plateaus.append({"target": float(targets[start]), "rate_g_h": abs(slope) * 3600.,
                         "mass_time_fit": monotone_quadratic(elapsed / 60., added),
                         "duration_s": float(elapsed[-1]), "observed": np.column_stack((elapsed[sample], added[sample])).tolist()})
    if not plateaus:
        raise ValueError("No positive pump setting had enough timed mass samples to fit a delivery rate.")
    return plateaus


def _rate_by_command(plateaus):
    groups = {}
    for run in plateaus:
        groups.setdefault(run["target"], []).append(run["rate_g_h"])
    return {target: float(np.mean(rates)) for target, rates in sorted(groups.items())}


def suggested_plateaus(plateaus):
    """Select an increasing set of rates for the initial review screen."""
    selected, last = set(), 0.
    for target, rate in _rate_by_command(plateaus).items():
        if rate > last:
            selected.add(target)
            last = rate
    return tuple(i for i, run in enumerate(plateaus) if run["target"] in selected)


def liquid_profile_calibration(log, plateaus, selected, device_type, command_unit, time_unit, density=None):
    selected = set(selected)
    runs = [run for i, run in enumerate(plateaus) if i in selected]
    path = log["path"]
    rates = _rate_by_command(runs)
    result = {"version": 1, "device_type": device_type, "input_unit": "g/h", "output_unit": command_unit, "origin_added": True,
              "format": "mass_log", "source_name": path.name, "source_file": str(path.resolve()),
              "sha256": sha256(path.read_bytes()).hexdigest(), "header_row": log["header"], "time_unit": time_unit,
              "columns": dict(zip(("time", "target", "mass"), log["columns"])),
              "points": [[0., 0.], *[[rate, setting] for setting, rate in rates.items()]],
              "plateau_rates": [[run["rate_g_h"], run["target"]] for run in runs],
              "dose_curves": [{**run, "included": index in selected} for index, run in enumerate(plateaus)]}
    return fit_liquid_flow(result, device_type, density)


def dose_duration(config, target, grams, *, preview=False):
    calibration = config.get("pump_calibration")
    if not calibration:
        raise ValueError("Load a liquid pump calibration before using a fixed pump setting.")
    limits = target_limits(config)
    if target <= 0 or (not preview and (calibration["output_unit"] != limits["unit"] or not target_allowed(target, limits))):
        raise ValueError(f"Choose a positive pump setting in {limits['unit']} that matches the calibration and device limits.")
    rate = float(delivered_flow(config, [target])[0])
    return grams * 3600. / rate, rate


def read_controller_target(filename):
    with Path(filename).open(encoding="utf-8-sig", newline="") as stream:
        values = [value.strip() for row in csv.reader(stream) for value in row if value.strip()]
    if len(values) != 1 or not np.isfinite(value := float(values[0])):
        raise ValueError("A controller export must contain one numeric pump setpoint.")
    return value


def calibration_equations(calibration):
    points = calibration["points"]
    lines = [calibration["source_name"]]
    if "rpm" in calibration:
        if model := calibration.get("seed_fit"):
            unit = calibration["time_unit"]
            scale = TIME_SCALES[unit]
            delay, rate, k = model["delay_s"] / scale, model["initial_rate_g_s"] * scale, model["k_s_inv"] * scale
            lines += [f"Dry seed at {calibration['rpm']:g} RPM. t is in {unit}, m is in g.",
                      f"0 ≤ t ≤ {delay:g} {unit}    m = 0"]
            if k:
                lines += [f"t > {delay:g} {unit}    m = {rate/k:.9g} × expm1({k:.9g} × (t − {delay:g}))",
                          f"Positive dose    t = {delay:g} + log1p(m / {rate/k:.9g}) / {k:.9g}"]
            else:
                lines += [f"t > {delay:g} {unit}    m = {rate:.9g} × (t − {delay:g}) (linear limit)",
                          f"Positive dose    t = {delay:g} + m / {rate:.9g}"]
            lines += ["Zero dose runs for zero time.",
                      f"Beyond {points[-1][0]/scale:g} {unit} or {points[-1][1]:g} g the fit extrapolates past the measurements."]
            return "\n".join(lines) + "\n"
        lines += [f"Dry seed at {calibration['rpm']:g} RPM", "t is elapsed pumping time in minutes and m is dry seed mass in grams.", ""]
        for (start, mass), (end, final_mass) in zip(points, points[1:]):
            start, end = start / 60., end / 60.
            slope = (final_mass - mass) / (end - start)
            formula = f"{mass:.9g} + {slope:.9g} × (t − {start:.9g})" if slope else f"{mass:.9g}"
            lines.append(f"{start:.9g} ≤ t ≤ {end:.9g} min    m = {formula}")
        lines += ["", "For a positive target dose, invert its segment to find t. A zero dose runs for zero minutes."]
    else:
        unit = calibration["output_unit"]
        lines += [f"Liquid pump setting u in {unit}", "t is elapsed pumping time in minutes, m is grams, and q is g/min.", ""]
        fit = calibration.get("flow_fit")
        if fit is not None:
            b, c = (value / 60. for value in fit)
            lines += [f"q(u) = 0 {b:+.9g} × u {c:+.9g} × u²    for 0 ≤ u ≤ {points[-1][1]:g} {unit}",
                      "Use q(u) × t for a constant pump command, or invert q(u) for the requested rate."]
            if density := calibration.get("density_g_ml"):
                lines += [f"Q(u) = 0 {b/density:+.9g} × u {c/density:+.9g} × u² mL/min at {density:g} g/mL."]
            else:
                lines += ["Enter liquid density to express this mass-flow fit in mL/min."]
        lines += ["", "Measured steady rates"]
        for rate, target in points:
            lines.append(f"{target:.9g} {unit}    q = {rate / 60.:.9g}")
        if fit is None:
            lines += ["", "Between settings (u₁, q₁) and (u₂, q₂)", "q(u) = q₁ + (q₂ − q₁) × (u − u₁) / (u₂ − u₁)"]
        lines += ["", "At a fixed setting, t = m / q(u). For a fixed duration, first calculate q = m / t."]
        mass_fits = [run for run in calibration.get("dose_curves", ()) if "mass_time_fit" in run]
        if mass_fits:
            lines += ["", "Mass versus time fits to individual logged runs (for review)"]
            for run in mass_fits:
                b, c = run["mass_time_fit"]
                note = "" if run["included"] else " · excluded"
                lines += [f"{run['target']:g} {unit}    m(t) = 0 {b:+.9g} × t {c:+.9g} × t²    "
                          f"for 0 ≤ t ≤ {run['duration_s']/60.:.6g} min{note}"]
        omitted = sorted({run["target"] for run in calibration.get("dose_curves", ()) if not run["included"]})
        if omitted:
            lines += ["", f"Runs excluded from the curve at these settings: {', '.join(f'{value:g}' for value in omitted)} {unit}."]
        if unit != "RPM":
            lines += ["", "These curves use the recorded volumetric settings. Ismatec execution requires a verified RPM curve."]
    return "\n".join(lines) + "\n"
