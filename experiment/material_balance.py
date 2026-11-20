"""Integrate physical pump delivery for charges and reservoir stock accounting."""
import numpy as np

from experiment.pump_calibration_files import delivered_flow


def pump_rate_g_s(config, commands, density):
    if config.get("pump_calibration"):
        if config["type"] == "ismatec" and config["pump_calibration"].get("output_unit") != "RPM":
            raise ValueError("Review the Ismatec calibration and apply a curve with measured RPM settings.")
        return delivered_flow(config, commands) / 3600.
    if (config["type"] == "ismatec" and config.get("simulated")
            and density is not None and np.isfinite(density) and density > 0):
        return np.asarray(commands) * density / 60.
    if config["type"] == "ismatec":
        raise ValueError("Load an RPM mass-flow calibration to convert Ismatec commands into grams.")
    raise ValueError("Load a mass-flow calibration to calculate delivered grams.")


def delivery_times(profile, config, *, start=0., end=None):
    if end is None:
        end = profile.points[-1].time_s
    x = np.array([p.time_s for p in profile.points])
    y = np.array([p.target for p in profile.points])
    times = list(x[(x > start) & (x < end)]) + [start, end]
    calibration = config.get("pump_calibration")
    if profile.interpolation == "linear":
        for a, b, ya, yb in zip(x, x[1:], y, y[1:]):
            if ya == yb:
                continue
            for command in ([0.] if not calibration else np.asarray(calibration["points"])[:, 1]):
                for target in (command, -command):
                    time = a + (target - ya) / (yb - ya) * (b - a)
                    if a < time < b and start < time < end:
                        times.append(time)
    return np.unique(times)


def pump_mass(profile, config, *, density=1., start=0., end=None):
    if end is None:
        end = profile.points[-1].time_s
    if end <= start:
        return 0.
    x = np.array([p.time_s for p in profile.points])
    y = np.array([p.target for p in profile.points])
    times = delivery_times(profile, config, start=start, end=end)
    try:
        if profile.interpolation == "step":
            indices = np.clip(np.searchsorted(x, times[:-1], side="right") - 1, 0, len(y) - 1)
            return float(np.dot(np.diff(times), pump_rate_g_s(config, y[indices], density)))
        rates = pump_rate_g_s(config, np.interp(times, x, y), density)
        if "flow_fit" in config.get("pump_calibration", {}):
            # Zero crossings are already split. Simpson's rule integrates a
            # quadratic rate over each linear RPM segment exactly.
            midpoints = (times[:-1] + times[1:]) / 2.
            mid_rates = pump_rate_g_s(config, np.interp(midpoints, x, y), density)
            return float(np.dot(np.diff(times), (rates[:-1] + 4.*mid_rates + rates[1:]) / 6.))
        return float(np.trapezoid(rates, times))
    except ValueError as error:
        raise ValueError(f"Pump {profile.device_id}: {error}") from error


def initial_charges(experiment, process, roles, configs, *, overrides=(), line_roles=None, issues=None):
    """A defined prep charge supplies that starting inventory; explicit model overrides win."""
    result = dict(process)
    begin = experiment.model_window[0] if experiment.model_window else 0.
    if begin <= 0:
        return result
    totals = {}
    volumes = {}
    seen = set()
    incomplete = set()
    for role, device in roles.items():
        role = (line_roles or {}).get(role, {}).get("role", role)
        if role not in {"AS", "solvent", "solution"}:
            continue
        if device in seen:
            continue
        seen.add(device)
        profile = next((p for p in experiment.profiles if p.device_id == device), None)
        if profile is None or not any(p.target for p in profile.points if p.time_s < begin):
            continue
        mass_key, density_key = ("AntiSolvMass", "AntiSolventDensity") if role == "AS" else ("SolvMass", "SolventDensity")
        density = configs[device].get("reservoir_density_g_ml") or result.get(density_key)
        try:
            mass = pump_mass(profile, configs[device], density=density, end=begin)
            if mass < 0:
                raise ValueError("A preparation withdrawal needs a measured starting mass override.")
        except ValueError as error:
            if mass_key in overrides:
                continue
            if issues is None:
                raise
            issues[mass_key] = str(error)
            incomplete.add(mass_key)
            continue
        totals[role] = totals.get(role, 0.) + mass
        if density is not None and np.isfinite(density) and density > 0 and volumes.get(role, 0.) is not None:
            volumes[role] = volumes.get(role, 0.) + mass / density
        else:
            volumes[role] = None
    if "AS" in totals and "AntiSolvMass" not in overrides and "AntiSolvMass" not in incomplete:
        result["AntiSolvMass"] = totals["AS"]
    if ("solvent" in totals or "solution" in totals) and "SolvMass" not in overrides and "SolvMass" not in incomplete:
        concentration = result.get("InitialConcentration")
        if "solution" in totals and concentration is None:
            message = "Enter the charging solution's initial concentration to calculate its solvent mass."
            if issues is None:
                raise ValueError(message)
            issues["SolvMass"] = message
            incomplete.add("SolvMass")
        else:
            result["SolvMass"] = totals.get("solvent", 0.) + totals.get("solution", 0.) * (1 - (concentration or 0.))
    if totals:
        result["PreparationCharges_g"] = totals
        result["PreparationVolumes_ml"] = {role: volume for role, volume in volumes.items() if volume is not None}
        if "InitialVolume_ml" not in overrides:
            result.pop("InitialVolume_ml", None)
    solvent_density, anti_density = (result.get(key) for key in ("SolventDensity", "AntiSolventDensity"))
    if (totals and not incomplete and "InitialVolume_ml" not in overrides and result.get("SolvMass") is not None
            and all(density is not None and np.isfinite(density) and density > 0 for density in (solvent_density, anti_density))):
        result["InitialVolume_ml"] = result["SolvMass"] / result["SolventDensity"] + result["AntiSolvMass"] / result["AntiSolventDensity"]
        if volumes.get("solution") is not None and "SolvMass" not in overrides:
            result["InitialVolume_ml"] += volumes["solution"] - totals["solution"] * (1 - (result.get("InitialConcentration") or 0.)) / result["SolventDensity"]
    if issues is not None:
        for key in incomplete:
            result.pop(key, None)
        if incomplete and "InitialVolume_ml" not in overrides:
            result.pop("InitialVolume_ml", None)
    return result
