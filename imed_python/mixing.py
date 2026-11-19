"""One scalar mixing input per experiment; never add profiles to the PBM RHS."""
from copy import deepcopy

import numpy as np
from scipy.integrate import quad


DEFAULT_MIXING = {"enabled": False, "power_number": 1., "diameter_m": .03,
                  "volume_mode": "fixed", "volume_ml": 100.}


def validate_mixing(settings):
    if not isinstance(settings, dict) or set(settings) - set(DEFAULT_MIXING):
        raise ValueError("Check the IMED average mixing settings.")
    result = {**DEFAULT_MIXING, **settings}
    # MATLAB structs commonly round-trip logical scalars as uint8 / int.
    if isinstance(result["enabled"], (bool, np.bool_, int, np.integer)) and result["enabled"] in (0, 1):
        result["enabled"] = bool(result["enabled"])
    if type(result["enabled"]) is not bool or result["volume_mode"] not in {"fixed", "charged"}:
        raise ValueError("Choose fixed volume or volume calculated from charged amounts.")
    for key in ("power_number", "diameter_m", "volume_ml"):
        value = result[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"IMED mixing {key} must be positive and finite.")
    return result


def trace_mean(points, duration, label, *, step=False):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 2
            or not np.isfinite(points).all() or np.any(np.diff(points[0]) <= 0)
            or np.any(points[1] < 0) or points[0, 0] > 1e-7 or points[0, -1] < duration - 1e-7):
        raise ValueError(f"{label} must cover the selected experiment period with finite nonnegative values.")
    knots = np.unique(np.r_[0., points[0, (points[0] > 0) & (points[0] < duration)], duration])
    if step:
        indices = np.clip(np.searchsorted(points[0], knots[:-1], side="right") - 1, 0, points.shape[1] - 1)
        return float(np.dot(np.diff(knots), points[1, indices]) / duration)
    return float(np.trapezoid(np.interp(knots, *points), knots) / duration)


def average_mixing(experiment, process, settings, *, mean_rpm=None, source="Experiment commands"):
    settings = validate_mixing(settings)
    result = deepcopy(process)
    result["MixingSettings"] = settings
    if not settings["enabled"]:
        result.update(PowerRatio=1., MixingSummary={"enabled": False, "power_ratio": 1.})
        return result
    duration = float(np.asarray(experiment["TemperatureData"])[0, -1])
    if duration <= 0:
        raise ValueError("Average mixing requires a positive experiment period.")
    if mean_rpm is None:
        mean_rpm = process.get("MixingSummary", {}).get("average_rpm")
    if mean_rpm is None or not np.isfinite(mean_rpm) or mean_rpm < 0:
        raise ValueError("Assign the main stirrer and include it in the experiment to calculate average mixing.")
    if settings["volume_mode"] == "fixed":
        volume_ml = settings["volume_ml"]
    else:
        # g / (g/cm³) = mL. Charges refer to the selected period's beginning.
        solvent, anti = float(process["SolvMass"]), float(process["AntiSolvMass"])
        density, anti_density = float(process["SolventDensity"]), float(process["AntiSolventDensity"])
        if min(solvent, anti) < 0 or min(density, anti_density) <= 0:
            raise ValueError("Charged-volume mixing needs nonnegative masses and positive liquid densities.")
        added = np.asarray(experiment.get("AddedAntisolventData", []), dtype=float)
        if added.size:
            added_mean = trace_mean(added, duration, "Added antisolvent mass")
        else:
            flow = np.asarray(experiment.get("FlowRateData", [[0., duration], [0., 0.]]), dtype=float)
            trace_mean(flow, duration, "Antisolvent feed")
            # Mean cumulative charge = integral_0^T (T-t) q(t) dt / T.
            knots = np.unique(np.r_[0., flow[0, (flow[0] > 0) & (flow[0] < duration)], duration])
            added_mean = sum(quad(lambda time: (duration - time) * np.interp(time, *flow), a, b)[0]
                             for a, b in zip(knots, knots[1:])) / duration
        initial_volume = process.get("InitialVolume_ml", solvent / density + anti / anti_density)
        volume_ml = initial_volume + added_mean / anti_density
    if not np.isfinite(volume_ml) or volume_ml <= 0:
        raise ValueError("The average liquid volume must be positive and finite.")
    epsilon = settings["power_number"] * (mean_rpm / 60.)**3 * settings["diameter_m"]**5 / (volume_ml * 1e-6)
    if not np.isfinite(epsilon):
        raise ValueError("The average mixing inputs produce a nonfinite power value.")
    result["PowerRatio"] = float(epsilon)
    result["MixingSummary"] = {"enabled": True, "average_rpm": float(mean_rpm), "average_volume_ml": float(volume_ml),
                               "epsilon_m2_s3": float(epsilon), "source": source,
                               "method": "Np * (average_RPM/60)^3 * diameter_m^5 / average_volume_m3"}
    return result
