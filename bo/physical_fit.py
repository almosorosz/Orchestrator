"""Fit a common, editable stage template to the recorded experiment profiles."""

from dataclasses import asdict

import numpy as np
from scipy.optimize import minimize

from .decoder import cooling_landmarks, fixed_horizon
from .physical import (
    Stage, apply_controls, control_values, grouped_controls, physical_controls,
    physical_profile, profile_stages, validate_stages,
)
from .profiles import make_profile


def read_spec(channel, spec, horizon=None):
    rows = tuple(Stage(**entry) for entry in spec["stages"])
    validate_stages(channel, rows, horizon)
    available = {control.key: control for control in (*grouped_controls(channel, rows),
                                                       *physical_controls(channel, rows))}
    keys = spec["controls"]
    if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Choose distinct physical controls per profile.")
    if any(key not in available for key in keys):
        raise ValueError("A selected physical control no longer applies to the stage list.")
    controls = tuple(available[key] for key in keys)
    # Check that the controls can be applied in combination before starting a fit.
    apply_controls(channel, rows, controls, control_values(rows, controls))
    return rows, controls


def choose_controls(channel, stages):
    available = {c.key: c for c in (*grouped_controls(channel, stages), *physical_controls(channel, stages))}
    if channel == "temperature":
        ramps = [i for i, stage in enumerate(stages) if stage.start != stage.end]
        requested = ["initial_delay", "final_temperature"]
        requested += [f"stage:{i}:duration" for i in ramps]
        requested.append("stage:0:start")
    else:
        active = [i for i, stage in enumerate(stages) if stage.start > 0 or stage.end > 0]
        requested = ["initial_delay", "on_time", "total_dose"]
        requested += [f"stage:{i}:duration" for i, stage in enumerate(stages)
                      if active and min(active) < i < max(active) and stage.start == stage.end == 0.]
    return [key for key in requested if key in available]


def trace_spec(channel, profile):
    """Initialize an editable recipe from the selected trace's phase changes."""
    stages = profile_stages(profile, channel=channel)
    if (channel == "temperature" and len(stages) > 1
            and abs(stages[-1].end - stages[-1].start) <= .15):
        stages = stages[:-1]
    return {"stages": [asdict(stage) for stage in stages],
            "controls": choose_controls(channel, stages)}


def default_spec(channel, profile):
    """Use a compact starting template for sampled traces, keeping commanded stages."""
    if profile.step or len(profile.time) <= 9:
        return trace_spec(channel, profile)
    duration = float(profile.time[-1])
    start, finish = float(profile.values[0]), float(profile.values[-1])
    if channel == "temperature":
        landmarks = cooling_landmarks(profile)
        if landmarks is None or not np.all(np.diff(landmarks) > 0):
            return trace_spec(channel, profile)
        first, middle, last = landmarks
        begin = max(0., first - .25 * (middle - first))
        end = min(duration, last + .25 * (last - middle))
        pivot = float(profile.sample(middle))
        stages = (Stage(middle - begin, start, pivot), Stage(end - middle, pivot, finish))
        if begin > 0:
            stages = (Stage(float(begin), start, start), *stages)
    else:
        if profile.total() <= 0.:
            return {"stages": [asdict(Stage(duration, 0., 0.))], "controls": []}
        active = np.flatnonzero(profile.values > .02 * np.max(profile.values))
        begin = float(profile.time[max(0, int(active[0]) - 1)])
        end = float(profile.time[min(len(profile.time) - 1, int(active[-1]) + 1)])
        rate = profile.total() / (end - begin)
        stages = (Stage(end - begin, rate, rate),)
        if begin > 0:
            stages = (Stage(begin, 0., 0.), *stages)
    return {"stages": [asdict(stage) for stage in stages],
            "controls": choose_controls(channel, stages)}


def initial_spec(channel, runs):
    profiles = [make_profile(data["points"], data["source"], step=data["step"])
                for run in runs if (data := run["profiles"].get(channel)) is not None]
    if not profiles:
        return None
    if channel == "feed":
        profiles = [profile for profile in profiles if np.min(profile.values) >= 0.]
        if not profiles:
            return None
        template = max(profiles, key=lambda profile: profile.total())
    else:
        template = max(profiles, key=lambda profile: np.ptp(profile.values))
    return default_spec(channel, template)


def measured_dose(profile, time):
    """Integrate the trace up to any query time using its exact stage breakpoints."""
    nodes = np.union1d(profile.time, time)
    rates = profile.sample(nodes)
    weights = rates[:-1] if profile.step else (rates[:-1] + rates[1:]) * .5
    return np.r_[0., np.cumsum(np.diff(nodes) * weights)][np.searchsorted(nodes, time)]


def fit_stage_controls(channel, rows, controls, source, horizon):
    """Estimate just the chosen controls, keeping the editor's other settings fixed."""
    source = fixed_horizon(source, horizon, stop=channel == "feed")
    nominal = control_values(rows, controls)
    time = np.linspace(0., horizon, max(181, 8 * len(rows) + 1))
    target = source.sample(time)
    value_scale = max(float(np.ptp(source.values)), 1. if channel == "temperature" else 1e-6)
    max_rate = max(float(np.max(source.values)), *(stage.start for stage in rows), *(stage.end for stage in rows))
    max_dose = max(float(source.total()), 1e-8) if channel == "feed" else 0.
    dose = measured_dose(source, time) if channel == "feed" else None
    bounds = []
    for control, value in zip(controls, nominal):
        if control.unit == "s":
            low, high = max(1e-6, horizon * 1e-7), horizon
        elif control.unit == "g":
            low, high = 0., 1.5 * max(max_dose, value)
        else:
            low = min(float(source.values.min()), value) if channel == "temperature" else 0.
            high = max_rate if channel == "temperature" else max_rate * 1.5
        bounds.append((min(low, value), max(high, value + 1e-9)))
    if not len(bounds):
        predicted = physical_profile(channel, rows, horizon)
        return nominal, float(np.sqrt(np.mean((predicted.sample(time) - target)**2)))
    lower, upper = np.asarray(bounds).T
    ranges = upper - lower

    def evaluate(fraction, *, error_only=False):
        values = lower + ranges * fraction
        try:
            stages = apply_controls(channel, rows, controls, values)
            decoded = physical_profile(channel, stages, horizon)
        except ValueError:
            return np.inf if error_only else 1e5
        rmse = float(np.sqrt(np.mean((decoded.sample(time) - target)**2)))
        if error_only:
            return rmse
        cost = rmse / value_scale
        if dose is not None:
            difference = (measured_dose(decoded, time) - dose) / max_dose
            cost += np.mean(np.abs(difference)) + abs(difference[-1])
        else:
            cost += .15 * abs(decoded.values[-1] - source.values[-1]) / value_scale
        return cost

    start = (nominal - lower) / ranges
    # A bounded derivative-free fit also handles discrete changes in feed rate.
    evaluation_budget = max(240, 60 * len(controls))
    result = minimize(evaluate, start, method="Powell", bounds=[(0., 1.)] * len(controls),
                      options={"maxfev": evaluation_budget, "xtol": .001, "ftol": .001})
    point = result.x if evaluate(result.x) < evaluate(start) else start
    return lower + ranges * point, evaluate(point, error_only=True)
