"""Fit a common, editable stage template to the recorded experiment profiles."""

from dataclasses import asdict
from math import fsum

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
        requested += [f"stage:{i}:end" for i in ramps[:-1]]
        requested.append("stage:0:start")
    else:
        active = [i for i, stage in enumerate(stages) if stage.start > 0 or stage.end > 0]
        requested = ["initial_delay", "on_time", "total_dose"]
        requested += [f"stage:{i}:duration" for i, stage in enumerate(stages)
                      if active and min(active) < i < max(active) and stage.start == stage.end == 0.]
    return [key for key in requested if key in available]


def trace_spec(channel, profile, *, exact=False):
    """Initialize an editable recipe from the selected trace's phase changes."""
    stages = profile_stages(profile, count=None if exact else 8, channel=channel)
    if len(stages) > 1:
        last = stages[-1]
        implicit_tail = (abs(last.end - last.start) <= (0. if exact else .15) if channel == "temperature"
                         else exact and last.start == last.end == 0.)
        if implicit_tail:
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


def _temperature_seed(rows, controls, source, nominal):
    boundaries = np.r_[0., np.cumsum([stage.duration_s for stage in rows])]
    moving = [index for index, stage in enumerate(rows) if stage.start != stage.end]
    observed = np.flatnonzero(np.abs(np.diff(source.values)) > 1e-8)
    vary_delay = any(control.kind == "delay" for control in controls)
    aligned = boundaries.copy()
    if moving and observed.size:
        first, last = moving[0], moving[-1] + 1
        actual_start = source.time[observed[0]]
        start = actual_start if first or vary_delay else 0.
        end = source.time[observed[-1] + 1]
        if first:
            aligned[:first] = boundaries[:first] * start / boundaries[first]
        ramp = boundaries[first:last + 1]
        aligned[first:last + 1] = start + (ramp - ramp[0]) / (ramp[-1] - ramp[0]) * (end - start)
        aligned[last + 1:] += end - boundaries[last]
        measured = profile_stages(source, count=None, channel="temperature")
        measured_ramps = [i for i, stage in enumerate(measured) if stage.start != stage.end]
        if measured_ramps and start == actual_start:
            begin, stop = measured_ramps[0], measured_ramps[-1] + 1
            if ([stage.start == stage.end for stage in rows[first:last]]
                    == [stage.start == stage.end for stage in measured[begin:stop]]):
                measured_times = np.r_[0., np.cumsum([stage.duration_s for stage in measured])]
                aligned[first:last + 1] = measured_times[begin:stop + 1]
    temperatures = source.sample(aligned)
    seed = nominal.copy()
    for column, control in enumerate(controls):
        index = control.stage
        if control.kind == "final":
            seed[column] = source.values[-1]
        elif control.kind == "delay" and observed.size:
            seed[column] = source.time[observed[0]]
        elif index is not None:
            if control.kind == "duration":
                seed[column] = aligned[index + 1] - aligned[index]
            elif control.kind == "start":
                seed[column] = temperatures[index]
            elif control.kind == "end":
                seed[column] = temperatures[index + 1]
    return seed


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
    best_cost, best_rmse, best_point = np.inf, np.inf, nominal

    def evaluate(fraction):
        nonlocal best_cost, best_rmse, best_point
        values = lower + ranges * fraction
        try:
            stages = apply_controls(channel, rows, controls, values)
            overage = fsum(stage.duration_s for stage in stages) - horizon
            if overage > np.spacing(horizon) * max(8, len(stages)):
                # Keep a slope toward shorter, feasible stages during the line search.
                return 1e4 + overage / horizon
            decoded = physical_profile(channel, stages, horizon)
        except ValueError:
            return 1e5
        rmse = float(np.sqrt(np.mean((decoded.sample(time) - target)**2)))
        cost = rmse / value_scale
        if dose is not None:
            difference = (measured_dose(decoded, time) - dose) / max_dose
            cost += np.mean(np.abs(difference)) + abs(difference[-1])
        else:
            cost += .15 * abs(decoded.values[-1] - source.values[-1]) / value_scale
        if cost < best_cost:
            best_cost, best_rmse, best_point = cost, rmse, values
        return cost

    start = (nominal - lower) / ranges
    evaluate(start)
    if channel == "temperature" and best_cost > 1e-10:
        seed = np.clip((_temperature_seed(rows, controls, source, nominal) - lower) / ranges, 0., 1.)
        for fraction in (1., .5, .25):
            evaluate(start + fraction * (seed - start))
    if best_cost <= 1e-10:
        return best_point, best_rmse
    # A bounded derivative-free fit also handles discrete changes in feed rate.
    evaluation_budget = max(240, 60 * len(controls))
    minimize(evaluate, (best_point - lower) / ranges, method="Powell", bounds=[(0., 1.)] * len(controls),
             options={"maxfev": evaluation_budget, "xtol": .001, "ftol": .001})
    if not np.isfinite(best_cost):
        raise ValueError("The selected controls could not fit a profile within the batch horizon.")
    return best_point, best_rmse
