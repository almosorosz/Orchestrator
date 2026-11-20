"""Compile fixed tubing operations into executable, recorded pump profiles."""
from dataclasses import asdict, replace
import math

import numpy as np

from experiment.material_balance import pump_rate_g_s
from experiment.models import ProfilePoint
from experiment.pump_calibration_files import OUTPUT_UNITS
from hardware_devices.limits import target_allowed, target_limits


def with_reservoir_densities(configs, topology):
    densities = {unit["code"]: unit.get("density_g_ml") for unit in topology.get("unit_operations", ())}
    return {key: {**config, "reservoir_density_g_ml": density}
            if (density := densities.get(config.get("connectivity", {}).get("source"))) is not None else config
            for key, config in configs.items()}


def validate_corrections(settings, config):
    """Missing settings keep the legacy behavior. An explicit empty object opts out."""
    if not isinstance(settings, dict):
        raise ValueError("Invalid pump corrections.")
    for key, label in (("prime_g", "Additional amount"), ("reverse_target", "Back-rotation speed"),
                       ("reverse_s", "Back-rotation duration")):
        value = settings.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{label} must be positive.")
    target = settings.get("reverse_target")
    if (target is None) != (settings.get("reverse_s") is None):
        raise ValueError("Back rotation needs both speed and duration.")
    if target is not None:
        limits = target_limits(config)
        if not target_allowed(-target, limits):
            raise ValueError(f"Back rotation at −{target:g} {limits['unit']} exceeds the pump's device limits.")
        if config.get("pump_calibration"):
            pump_rate_g_s(config, [-target], None)
    if settings.get("prime_g") is not None and not config.get("pump_calibration"):
        if config["type"] != "ismatec" or not config.get("simulated"):
            raise ValueError("Load the liquid mass-flow calibration to convert the line-fill grams into pumping time.")
    return settings


def _edge(time, span):
    """A short transition preserves discontinuities in an otherwise linear profile."""
    return time - min(1e-6, span / 4.)


def _schedule(profile, duration):
    first = next(i for i, point in enumerate(profile.points) if point.target > 0)
    begin = profile.points[first].time_s
    if first and profile.interpolation == "linear":
        before, after = profile.points[first - 1:first + 1]
        begin = before.time_s + (-before.target) * (after.time_s - before.time_s) / (after.target - before.target)
    last = max(i for i, point in enumerate(profile.points) if point.target != 0)
    finish = profile.points[last + 1].time_s if last + 1 < len(profile.points) else duration
    return begin, finish, profile.points[first].target


def _execution_profile(profile, config, shift, nominal_end, end, actions):
    points = {p.time_s + shift: p.target for p in profile.points}
    is_pump = config["type"] in OUTPUT_UNITS
    if shift:
        initial = profile.points[0].target
        points[0.] = 0. if is_pump or config["type"] == "nidaq_valve" else initial
        points.setdefault(shift, initial)
        if is_pump:
            points[_edge(shift, shift)] = 0.
    if end > nominal_end:
        points[nominal_end] = profile.points[-1].target
        points[end] = 0. if is_pump else profile.points[-1].target
        if is_pump:
            points[nominal_end + min(1e-6, (end - nominal_end) / 4.)] = 0.
    for kind in ("prime", "reverse"):
        if kind not in actions:
            continue
        action = actions[kind]
        start, stop = action["start_s"], action["end_s"]
        restore = points.get(stop, 0.) if kind == "prime" else 0.
        if start > 0 and profile.interpolation == "linear":
            times = sorted(points)
            previous = max((time for time in times if time < start), default=0.)
            edge = _edge(start, start - previous)
            points[edge] = float(np.interp(edge, times, [points[time] for time in times]))
        points = {time: value for time, value in points.items() if not start <= time <= stop}
        points[start] = action["target"]
        if profile.interpolation == "linear":
            points[_edge(stop, stop - start)] = action["target"]
        points[stop] = restore
    return replace(profile, points=tuple(ProfilePoint(time, value) for time, value in sorted(points.items())))


def apply_pump_corrections(experiment, configs, *, window=None):
    if experiment.pump_adjustments:
        return experiment
    duration = max(profile.points[-1].time_s for profile in experiment.profiles) or 5.
    model_window = window or experiment.model_window
    model_start, model_end = model_window or (0., duration)
    adjustments = {}
    lead = 0.
    for profile in experiment.profiles:
        key = profile.device_id
        config = configs.get(key, {})
        settings = config.get("pump_corrections")
        if config.get("type") not in OUTPUT_UNITS or not settings or not any(p.target > 0 for p in profile.points):
            continue
        try:
            validate_corrections(settings, config)
            begin, finish, prime_target = _schedule(profile, duration)
            actions = {}
            if mass := settings.get("prime_g"):
                rate = pump_rate_g_s(config, [prime_target], 1.)[0]
                start = begin - mass / rate
                if any(point.target < 0 and point.time_s <= begin for point in profile.points):
                    raise ValueError("The profile reverses before its first delivery. Move that reverse after the dose to use line fill.")
                lead = max(lead, -start)
                actions["prime"] = {"start_s": start, "end_s": begin, "target": prime_target, "mass_g": mass}
            if target := settings.get("reverse_target"):
                seconds = settings["reverse_s"]
                start = model_end if finish < model_end and finish + seconds > model_start else finish
                actions["reverse"] = {"start_s": start, "end_s": start + seconds, "target": -target,
                                      "return_limit_g": settings.get("prime_g") or 0.}
                if config.get("pump_calibration") or (config["type"] == "ismatec" and config.get("reservoir_density_g_ml")):
                    rate = pump_rate_g_s(config, [-target], config.get("reservoir_density_g_ml"))[0]
                    actions["reverse"]["mass_g"] = float(rate * seconds)
            if actions:
                adjustments[key] = {**actions, "nominal_profile": asdict(profile)}
        except ValueError as error:
            raise ValueError(f"{experiment.name}, pump {key}: {error}") from error
    if not adjustments:
        return experiment
    end = duration + lead
    for item in adjustments.values():
        for kind in ("prime", "reverse"):
            if kind in item:
                action = item[kind]
                action["start_s"] = float(action["start_s"] + lead)
                action["end_s"] = float(action["end_s"] + lead)
                end = max(end, action["end_s"])
    metadata = {"lead_in_s": float(lead), "nominal_end_s": float(duration + lead), "pumps": adjustments}
    profiles = tuple(_execution_profile(profile, configs[profile.device_id], lead, duration + lead, end,
                                       adjustments.get(profile.device_id, {})) for profile in experiment.profiles)
    return replace(experiment, profiles=profiles, pump_adjustments=metadata,
                   model_window=(model_start + lead, model_end + lead),
                   control_modes=tuple(replace(event, time_s=event.time_s + lead) for event in experiment.control_modes),
                   run_stages=tuple(replace(stage, start_s=stage.start_s + lead, end_s=stage.end_s + lead)
                                    for stage in experiment.run_stages))


def corrected_campaign(campaign, configs, windows=None):
    return replace(campaign, experiments=tuple(apply_pump_corrections(experiment, configs,
                            window=(windows or {}).get(experiment.experiment_id)) for experiment in campaign.experiments))


def progress_fraction(action, time):
    return np.clip((time - action["start_s"]) / (action["end_s"] - action["start_s"]), 0., 1.)


def modeled_frame(frame, adjustments):
    """Recover the unadjusted timeline for the built-in PBM simulator."""
    if not adjustments:
        return frame
    shift = adjustments["lead_in_s"]
    selected = frame["time_s"].between(shift, adjustments["nominal_end_s"])
    result = frame.loc[selected].copy()
    result["time_s"] -= shift
    time = result["time_s"].to_numpy()
    for key, actions in adjustments["pumps"].items():
        nominal = actions["nominal_profile"]
        x = np.array([point["time_s"] for point in nominal["points"]])
        y = np.array([point["target"] for point in nominal["points"]])
        if nominal["interpolation"] == "step":
            result[key] = y[np.clip(np.searchsorted(x, time, side="right") - 1, 0, len(y) - 1)]
        else:
            result[key] = np.interp(time, x, y)
    return result
