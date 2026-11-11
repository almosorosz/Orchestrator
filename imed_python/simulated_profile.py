"""Inputs for new PBM predictions built from a recorded DoE experiment."""

import numpy as np

from .campaign_import import seed_distribution
from .solver import model_parameters, profile


FACTOR_GROUPS = {
    "temperature": {"t", "t_diss", "t_0", "t_h_0", "t_cool", "t_final", "t_h_cool", "t_eq", "arr"},
    "feed": {"as", "as_initial", "t_h_as_initial", "as_am", "as_add_time", "t_h_as", "arr"},
    "seed": {"seed", "t_h_seed"},
    "stirring": {"stirring_rate"},
}
STIRRING_FIELDS = ("StirringRPM", "StirringRate", "stirring_rate")
PROFILE_GROUPS = {
    "temperature": ("temperature",),
    "feed": ("feed", "initial_antisolvent"),
    "seed": ("seed_mass", "seed_percentiles"),
    "stirring": ("stirring_rpm", "reference_rpm"),
}


def power_dependent(model):
    parameters = model_parameters(model)
    # Secondary nucleation and agglomeration both use PowerRatio raised to an exponent.
    return any(np.isfinite(parameters[rate]) and np.isfinite(parameters[exponent])
               and parameters[exponent] != 0 for rate, exponent in ((5, 11), (24, 27)))


def _varies(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return bool(values.size > 1 and not np.allclose(values, values[0]))


def _trace_varies(experiments, key, *, default=None):
    traces = [np.asarray(item.get(key, []), dtype=float).reshape(2, -1) for item in experiments]
    reference, *others = traces
    for trace in others:
        if not reference.size and not trace.size:
            continue
        time = np.union1d(reference[0], trace[0])
        if default is None and (not reference.size or not trace.size
                                or not np.isclose(reference[0, -1], trace[0, -1])):
            return True
        first = profile(reference.squeeze(), default=default)
        second = profile(trace.squeeze(), default=default)
        if not np.allclose(first(time), second(time)):
            return True
    return False


def template_rpm(process):
    for field in STIRRING_FIELDS:
        value = np.asarray(process.get(field, []), dtype=float)
        if value.size == 1:
            rpm = value.item()
            if np.isfinite(rpm) and rpm > 0:
                return rpm
    return np.nan


def editable_groups(case, model_index, saved_profile=None):
    """Use declared DoE factors, with comparisons across runs for older cases."""
    workflow = case.get("PythonWorkflow", {})
    factors = workflow.get("doe_factors", case.get("active_doe_factors", workflow.get("profile_doe_factors")))
    if factors is not None:
        selected = {str(value).strip().casefold() for value in np.asarray(factors).reshape(-1)}
        groups = {group for group, keys in FACTOR_GROUPS.items() if selected & keys}
    else:
        experiments, processes = case["ExpData"], case["ProcInf"]
        groups = set()
        if len(experiments) > 1:
            if _trace_varies(experiments, "TemperatureData"):
                groups.add("temperature")
            if (_trace_varies(experiments, "FlowRateData", default=0.)
                    or _varies([p.get("AntiSolvMass", np.nan) for p in processes])):
                groups.add("feed")
            seeded = [exp for exp, proc in zip(experiments, processes) if proc.get("SeedMass", 0) > 0]
            if (_varies([p.get("SeedMass", np.nan) for p in processes])
                    or (len(seeded) > 1 and _trace_varies(seeded, "InitialCSDsData", default=0.))):
                groups.add("seed")
            if _varies([template_rpm(p) for p in processes]):
                groups.add("stirring")
    # A saved recipe can still be edited after its original DoE is closed.
    groups.update(group for group, fields in PROFILE_GROUPS.items()
                  if any(field in (saved_profile or {}) for field in fields))
    if not power_dependent(case["ModelParameter"][model_index]):
        groups.discard("stirring")
    return groups


def remember_active_doe(case):
    if "active_doe_factors" in case:
        case["PythonWorkflow"]["profile_doe_factors"] = np.asarray(case["active_doe_factors"], dtype=object)


def stirring_power_ratio(reference_power, rpm, reference_rpm):
    values = np.array([reference_power, rpm, reference_rpm], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Enter positive stirring speeds and use a template with a positive power ratio.")
    with np.errstate(over="ignore", under="ignore"):
        power = values[0] * (values[1] / values[2])**3
    if not np.isfinite(power) or power <= 0:
        raise ValueError("The stirring speeds produce an invalid power ratio.")
    return float(power)


def _knots(data, label):
    points = np.asarray(data, dtype=float)
    if (points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 2
            or not np.isfinite(points).all() or points[0, 0] != 0
            or np.any(np.diff(points[0]) <= 0)):
        raise ValueError(f"{label} needs at least two finite points with times starting at 0 and increasing.")
    return points


def feed_for_duration(data, end):
    """Represent a feed on the run horizon, holding its endpoint rates."""
    points = np.asarray(data, dtype=float).reshape(2, -1)
    time = np.r_[0., points[0, (points[0] > 0) & (points[0] < end)], end]
    rates = np.interp(time, *points) if points.size else np.zeros(time.size)
    return np.vstack((time, rates))


def prediction_inputs(case, experiment_index, simulated_profile=None):
    """Prepare prediction-only inputs without altering the fitted DoE case."""
    template = case["ExpData"][experiment_index]
    base_process = case["ProcInf"][experiment_index]
    if not simulated_profile:
        return template, base_process

    recipe = simulated_profile
    template_temperature = np.asarray(template["TemperatureData"], dtype=float).reshape(2, -1)
    temperature = template_temperature
    if "temperature" in recipe:
        temperature = _knots(recipe["temperature"], "Temperature")
        if np.any(temperature[1] <= -273.15):
            raise ValueError("Temperature must be above absolute zero.")
    end = temperature[0, -1]
    flow = template.get("FlowRateData", [])
    if "feed" in recipe:
        flow = _knots(recipe["feed"], "Antisolvent feed")
        if flow[0, -1] > end or np.any(flow[1] < 0):
            raise ValueError("Use nonnegative antisolvent rates and times within the simulation duration.")
        flow = feed_for_duration(flow, end)
    elif end != template_temperature[0, -1]:
        flow = feed_for_duration(flow, end)

    concentration = (template["InitialConcentration"] if "InitialConcentration" in template
                     else np.asarray(template["ConcentrationData"]).reshape(2, -1)[1, 0])
    seed_csd = template.get("InitialCSDsData", [])
    process = dict(base_process)
    if "initial_antisolvent" in recipe:
        antisolvent = float(recipe["initial_antisolvent"])
        if not np.isfinite(antisolvent) or antisolvent < 0:
            raise ValueError("Initial antisolvent mass must be finite and nonnegative.")
        if process["SolvMass"] + antisolvent <= 0:
            raise ValueError("The starting liquid mass must be positive.")
        process["AntiSolvMass"] = antisolvent
    if "seed_mass" in recipe:
        seed_mass = float(recipe["seed_mass"])
        if not np.isfinite(seed_mass) or seed_mass < 0:
            raise ValueError("Initial seed mass must be finite and nonnegative.")
        process["SeedMass"] = seed_mass
    if "seed_percentiles" in recipe:
        sizes = np.asarray(recipe["seed_percentiles"], dtype=float)
        if sizes.shape != (3,) or not np.isfinite(sizes).all():
            raise ValueError("Enter finite seed D10, D50 and D90 values.")
        seed_csd = seed_distribution(sizes)
    if process["SeedMass"] > 0 and not np.size(seed_csd):
        raise ValueError("Enter seed D10, D50 and D90 for a positive seed mass.")
    if "stirring_rpm" in recipe:
        process["PowerRatio"] = stirring_power_ratio(
            process["PowerRatio"], recipe["stirring_rpm"], recipe["reference_rpm"])
    return {"TemperatureData": temperature, "FlowRateData": flow,
            "InitialConcentration": concentration, "InitialCSDsData": seed_csd}, process
