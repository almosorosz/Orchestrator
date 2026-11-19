"""Integrate physical pump delivery for charges and reservoir stock accounting."""
import numpy as np

from experiment.pump_calibration_files import delivered_flow


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
    calibration = config.get("pump_calibration")
    def rate(commands):
        if calibration:
            return delivered_flow(config, commands) / 3600.
        if config["type"] == "ismatec":
            if density is None or not np.isfinite(density) or density <= 0:
                raise ValueError("Enter a positive reservoir density to convert mL/min into grams.")
            return np.asarray(commands) * density / 60.
        raise ValueError(f"Load a mass-flow calibration for pump {profile.device_id} to calculate delivered grams.")
    if profile.interpolation == "step":
        indices = np.clip(np.searchsorted(x, times[:-1], side="right") - 1, 0, len(y) - 1)
        return float(np.dot(np.diff(times), rate(y[indices])))
    return float(np.trapezoid(rate(np.interp(times, x, y)), times))


def initial_charges(experiment, process, roles, configs, *, overrides=(), line_roles=None):
    """A defined prep charge supplies that starting inventory; explicit model overrides win."""
    result = dict(process)
    begin = experiment.model_window[0] if experiment.model_window else 0.
    if begin <= 0:
        return result
    totals = {}
    volumes = {}
    seen = set()
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
        density = configs[device].get("reservoir_density_g_ml", result["AntiSolventDensity"] if role == "AS" else result["SolventDensity"])
        mass = pump_mass(profile, configs[device], density=density, end=begin)
        if mass < 0:
            raise ValueError("A preparation withdrawal cannot define the starting inventory; enter its final masses as experiment overrides.")
        totals[role] = totals.get(role, 0.) + mass
        volumes[role] = volumes.get(role, 0.) + mass / density
    if "AS" in totals and "AntiSolvMass" not in overrides:
        result["AntiSolvMass"] = totals["AS"]
    if ("solvent" in totals or "solution" in totals) and "SolvMass" not in overrides:
        concentration = result.get("InitialConcentration")
        if "solution" in totals and concentration is None:
            raise ValueError("Enter the charging solution's initial concentration to convert its mass into solvent and dissolved solute.")
        result["SolvMass"] = totals.get("solvent", 0.) + totals.get("solution", 0.) * (1 - (concentration or 0.))
    if totals:
        result["PreparationCharges_g"] = totals
        result["PreparationVolumes_ml"] = volumes
        result["InitialVolume_ml"] = result["SolvMass"] / result["SolventDensity"] + result["AntiSolvMass"] / result["AntiSolventDensity"]
        if "solution" in totals and "SolvMass" not in overrides:
            result["InitialVolume_ml"] += volumes["solution"] - totals["solution"] * (1 - (result.get("InitialConcentration") or 0.)) / result["SolventDensity"]
    return result
