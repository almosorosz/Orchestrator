"""Propagate joint parameter covariance and operating-condition uncertainty."""
from functools import partial

import numpy as np
from scipy.stats import binomtest, rankdata

from .candidate_workers import candidate_map
from .comparison import percentile_history
from .distribution import d_values
from .estimation import TRIAL_ERRORS, series
from .parameter_labels import PARAMETERS
from .session import history, signature, unpack_fit, unpack_intervals
from .simulated_profile import editable_groups, prediction_inputs, stirring_power_ratio
from .solver import model_parameters, simulate
from .uncertainty import DISCRETE_PARAMETERS, intervals_from_covariance


CONDITIONS = (
    ("temperature_offset", "Temperature offset (°C)", -2., 2.),
    ("feed_factor", "AS feed multiplier", .9, 1.1),
    ("seed_factor", "Seed mass multiplier", .9, 1.1),
    ("concentration_factor", "Initial concentration multiplier", .95, 1.05),
    ("duration_factor", "Profile duration multiplier", .9, 1.1),
    ("stirring_factor", "Stirring speed multiplier", .9, 1.1),
)
ERROR_LEVELS = (95., 99.)


def model_intervals(case, model_index):
    """Find covariance at the current values, including pre-PE local CIs."""
    model = case["ModelParameter"][model_index]
    current = model_parameters(model)
    active = np.flatnonzero(np.asarray(model["paraset"])[:, 0] == 1)
    contexts = {
        "parameter_uncertainty": signature(case, model_index, include_weights=False),
        "parameter_estimation": signature(case, model_index, include_values=False),
    }
    for step in reversed(history(case)):
        if (int(step["model_index"]) != model_index
                or step["action"] not in contexts
                or step["signature"] != contexts[step["action"]]):
            continue
        result = (unpack_intervals(step["result"].get("intervals"))
                  if step["action"] == "parameter_uncertainty"
                  else unpack_fit(step["result"]).intervals)
        if result is not None and np.array_equal(current[result.indices], result.values):
            return result
    covariance = np.asarray(model.get("covmat", []), dtype=float)
    if covariance.shape == (len(active), len(active)) and active.size:
        positions = np.flatnonzero(~np.isin(active, DISCRETE_PARAMETERS)
                                   & (np.asarray(model["paraset"])[active, 3]
                                      > np.asarray(model["paraset"])[active, 2]))
        if positions.size:
            degrees = max(1, np.size(model.get("resid", [])) - len(active))
            return intervals_from_covariance(active[positions], current[active[positions]],
                                             covariance[np.ix_(positions, positions)], degrees)
    raise ValueError("Calculate parameter CIs for this model at its current values first, "
                     "or apply a fitted model with covariance.")


def joint_parameter_draws(intervals, count, rng):
    covariance = np.asarray(intervals.covariance, dtype=float)
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=scale*1e-12):
        raise ValueError("The parameter covariance must be symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T)/2)
    if eigenvalues[0] < -scale*1e-10:
        raise ValueError("The parameter covariance must be positive semidefinite.")
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    return intervals.values + rng.standard_normal((count, len(intervals.indices))) @ root.T


def error_limits(errors, levels=ERROR_LEVELS):
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.full(len(levels), np.nan)
    # MATLAB prctile's midpoint plotting positions are NumPy's Hazen method.
    return np.percentile(np.abs(values), levels, method="hazen")


def prediction_errors(result, *, scope="all_times"):
    """Signed size errors at observed times, retaining failed draws as NaNs."""
    if scope not in ("all_times", "final"):
        raise ValueError("Choose all measured times or final measured sizes.")
    time = np.asarray(result["time"], dtype=float).reshape(-1)
    predictions = np.asarray(result["predictions"], dtype=float).reshape(-1, 3, len(time))
    nominal = np.asarray(result["nominal"]["percentiles"], dtype=float).reshape(3, len(time))
    valid = np.asarray(result["valid"], dtype=bool).reshape(-1)
    if valid.sum() != len(predictions):
        raise ValueError("The saved predictions do not match the successful draw count.")
    errors = {}
    for column, name in enumerate(("D10", "D50", "D90")):
        coordinates = np.asarray(result["observation_time"][name], dtype=float).reshape(-1)
        measured = np.asarray(result["observed"][name], dtype=float).reshape(-1)
        usable = (np.isfinite(coordinates) & np.isfinite(measured) & (measured > 0)
                  & (coordinates >= time[0]) & (coordinates <= time[-1]))
        coordinates, measured = coordinates[usable], measured[usable]
        order = np.argsort(coordinates)
        coordinates, measured = coordinates[order], measured[order]
        if scope == "final":
            coordinates, measured = coordinates[-1:], measured[-1:]
        sampled = np.full((len(valid), len(coordinates)), np.nan)
        for row, trace in zip(np.flatnonzero(valid), predictions):
            sampled[row] = np.interp(coordinates, time, trace[column])
        baseline = np.interp(coordinates, time, nominal[column])
        absolute, nominal_absolute = sampled-measured, baseline-measured
        relative = np.divide(100*absolute, sampled, out=np.full_like(absolute, np.nan), where=sampled > 0)
        nominal_relative = np.divide(100*nominal_absolute, baseline, out=np.full_like(baseline, np.nan),
                                     where=baseline > 0)
        errors[name] = {"time": coordinates, "absolute": absolute, "relative": relative,
                        "nominal_absolute": nominal_absolute, "nominal_relative": nominal_relative}
    return errors


def assess_reliability(errors, threshold_percent):
    """Invert the empirical absolute-error CDF and assess complete model draws."""
    if not np.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError("Enter a finite, nonnegative error threshold.")
    responses = {}
    draw_count = np.asarray(errors["D10"]["relative"]).shape[0]
    for name in ("D10", "D50", "D90"):
        relative = np.asarray(errors[name]["relative"], dtype=float)
        within = np.isfinite(relative) & (np.abs(relative) <= threshold_percent)
        observations = relative.shape[1]
        passing_draws = within.all(axis=1) if observations else np.zeros(draw_count, dtype=bool)
        count, successes = relative.size, int(within.sum())
        responses[name] = {
            "observation_count": observations, "comparison_count": count,
            "comparison_successes": successes,
            "comparison_reliability": 100*successes/count if count else np.nan,
            "passing_draws": passing_draws, "passing_draw_count": int(passing_draws.sum()),
            "draw_reliability": 100*passing_draws.mean() if count else np.nan,
        }
    joint_available = draw_count > 0 and all(row["observation_count"] for row in responses.values())
    joint_passes = np.logical_and.reduce([row["passing_draws"] for row in responses.values()])
    joint_count = int(joint_passes.sum())
    sampling_interval = np.full(2, np.nan)
    if joint_available:
        interval = binomtest(joint_count, draw_count).proportion_ci(confidence_level=.95, method="exact")
        sampling_interval = 100*np.array([interval.low, interval.high])
    return {"threshold_percent": float(threshold_percent), "draw_count": draw_count,
            "responses": responses, "joint_available": joint_available,
            "joint_passing_draws": joint_passes, "joint_passing_draw_count": joint_count,
            "joint_reliability": 100*joint_count/draw_count if joint_available else np.nan,
            "joint_sampling_interval": sampling_interval}


def sample_intervals(values):
    """Pointwise central 95% and 99% intervals across successful model draws."""
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    usable = np.isfinite(values).any(axis=0)
    bands = np.full((5, values.shape[1]), np.nan)
    if usable.any():
        bands[:, usable] = np.nanpercentile(values[:, usable], [.5, 2.5, 50, 97.5, 99.5],
                                            axis=0, method="hazen")
    return dict(zip(("lower99", "lower95", "median", "upper95", "upper99"), bands))


def observed_percentiles(experiment, end):
    times, measured = [], []
    final = np.asarray(experiment.get("FinalD10D50D90", []), dtype=float).reshape(-1)
    if final.size != 3 and np.size(experiment.get("FinalCSDsData", [])):
        csd = series(experiment["FinalCSDsData"], "FinalCSDsData")
        final = d_values(*csd)
    for i, name in enumerate(("D10Data", "D50Data", "D90Data")):
        points = np.asarray(experiment.get(name, []), dtype=float)
        if points.size:
            points = series(points, name)
            valid = (points[0] >= 0) & (points[0] <= end) & (points[1] > 0)
            t, y = points[:, valid]
        elif final.size == 3 and np.isfinite(final[i]) and final[i] > 0:
            t, y = np.array([end]), np.array([final[i]])
        else:
            t = y = np.empty(0)
        times.append(t)
        measured.append(y)
    return times, measured


def fbrm_calibration(nominal, experiment):
    """Fit a fixed proportional conversion from PBM number density to FBRM counts."""
    measured = np.asarray(experiment.get("CountsData", []), dtype=float)
    if not measured.size:
        return {"scale": np.nan, "observed": np.empty((2, 0)), "scaled": False}
    measured = series(measured, "CountsData")
    valid = ((measured[0] >= nominal["time"][0]) & (measured[0] <= nominal["time"][-1])
             & (measured[1] >= 0))
    measured = measured[:, valid]
    predicted = np.interp(measured[0], nominal["time"], nominal["counts"])
    denominator = predicted @ predicted
    if not np.isfinite(denominator) or denominator <= 0:
        return {"scale": np.nan, "observed": measured, "scaled": False}
    scale = max(0., float(predicted @ measured[1] / denominator))
    return {"scale": scale, "observed": measured, "scaled": True}


def _predict(parameters, experiment, process, solubility, settings, *, cancelled=None):
    count, concentration, csd, state = simulate(parameters, experiment, process, solubility,
                                              settings, return_history=True, cancelled=cancelled)
    return {"time": count[0], "counts": count[1], "concentration": concentration[1],
            "percentiles": percentile_history(csd, state, process, settings, cancelled=cancelled)}


def _trial(candidate, *, baseline, indices, condition_keys, experiment, process, solubility,
           settings, cancelled=None):
    parameters = baseline.copy()
    parameters[indices] = candidate[:len(indices)]
    conditions = dict(zip(condition_keys, candidate[len(indices):]))
    experiment, process = dict(experiment), dict(process)
    temperature = np.asarray(experiment["TemperatureData"], dtype=float).copy()
    duration = conditions.get("duration_factor", 1.)
    temperature[0] *= duration
    temperature[1] += conditions.get("temperature_offset", 0.)
    experiment["TemperatureData"] = temperature
    if np.size(experiment.get("FlowRateData", [])):
        flow = np.asarray(experiment["FlowRateData"], dtype=float).reshape(2, -1).copy()
        flow[0] *= duration
        flow[1] *= conditions.get("feed_factor", 1.)
        experiment["FlowRateData"] = flow
    process["SeedMass"] *= conditions.get("seed_factor", 1.)
    initial = (experiment["InitialConcentration"] if "InitialConcentration" in experiment
               else np.asarray(experiment["ConcentrationData"]).reshape(2, -1)[1, 0])
    experiment["InitialConcentration"] = initial * conditions.get("concentration_factor", 1.)
    try:
        if "stirring_factor" in conditions:
            process["PowerRatio"] = stirring_power_ratio(process["PowerRatio"], conditions["stirring_factor"], 1.)
        return _predict(parameters, experiment, process, solubility, settings, cancelled=cancelled), ""
    except TRIAL_ERRORS as error:
        return None, str(error)


def partial_rank_correlations(inputs, outputs):
    """PRCCs with other varying inputs controlled through rank regression."""
    inputs, outputs = np.asarray(inputs), np.asarray(outputs)
    result = np.full((inputs.shape[1], outputs.shape[1]), np.nan)
    if not len(inputs):
        return result
    complete_inputs = np.isfinite(inputs).all(axis=1)
    if not (complete_inputs.all() and np.isfinite(outputs).all()):
        for column in range(outputs.shape[1]):
            usable = complete_inputs & np.isfinite(outputs[:, column])
            result[:, column] = partial_rank_correlations(inputs[usable], outputs[usable, column:column+1])[:, 0]
        return result
    varying = np.flatnonzero(np.ptp(inputs, axis=0) > 0)
    if len(inputs) <= len(varying) + 2:
        return result
    ranked = rankdata(inputs[:, varying], axis=0)
    ranked_outputs = rankdata(outputs, axis=0)
    for position, column in enumerate(varying):
        controls = np.column_stack((np.ones(len(inputs)), np.delete(ranked, position, axis=1)))
        if np.linalg.matrix_rank(controls) != controls.shape[1]:
            continue
        x = ranked[:, position] - controls @ np.linalg.lstsq(controls, ranked[:, position], rcond=None)[0]
        y = ranked_outputs - controls @ np.linalg.lstsq(controls, ranked_outputs, rcond=None)[0]
        denominator = np.linalg.norm(x) * np.linalg.norm(y, axis=0)
        usable = denominator > 1e-10 * np.linalg.norm(ranked[:, position]) * np.linalg.norm(ranked_outputs, axis=0)
        result[column, usable] = np.clip((x @ y[:, usable])/denominator[usable], -1, 1)
    return result


def prcc_diagnostics(inputs, outputs):
    inputs, outputs = np.asarray(inputs), np.asarray(outputs)
    input_spans = np.ptp(inputs, axis=0) if len(inputs) else np.zeros(inputs.shape[1])
    complete = np.isfinite(inputs).all(axis=1)
    counts = np.zeros(outputs.shape[1], dtype=int)
    output_spans = np.full(outputs.shape[1], np.nan)
    for column in range(outputs.shape[1]):
        usable = complete & np.isfinite(outputs[:, column])
        counts[column] = usable.sum()
        if counts[column]:
            output_spans[column] = np.ptp(outputs[usable, column])
    return {"input_spans": input_spans, "output_spans": output_spans,
            "draw_counts": counts, "minimum_draws": int((input_spans > 0).sum())+3}


def propagate(case, model_index, experiment_index, *, samples=100, seed=0, use_parameters=True,
              conditions=None, workers=1, simulated_profile=None, cancelled=None, progress=None,
              draw_progress=None):
    if not 2 <= samples <= 10000:
        raise ValueError("Use between 2 and 10000 samples.")
    if progress is not None:
        progress("Preparing sample inputs…")
    conditions = conditions or {}
    if not use_parameters and not conditions:
        raise ValueError("Select parameter uncertainty, operating conditions, or both.")
    if ("stirring_factor" in conditions or "stirring_rpm" in (simulated_profile or {})):
        if "stirring" not in editable_groups(case, model_index, simulated_profile):
            raise ValueError("Stirring requires a DoE stirring factor and a power-dependent kinetic term.")
    model = case["ModelParameter"][model_index]
    baseline = model_parameters(model)
    rng = np.random.default_rng(seed)
    indices = np.empty(0, dtype=int)
    covariance = np.empty((0, 0))
    parameter_draws = np.empty((samples, 0))
    if use_parameters:
        intervals = model_intervals(case, model_index)
        indices, covariance = intervals.indices, intervals.covariance
        parameter_draws = joint_parameter_draws(intervals, samples, rng)
    keys = list(conditions)
    known = {item[0] for item in CONDITIONS}
    for key, bounds in conditions.items():
        if (key not in known or len(bounds) != 2 or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1] or (key != "temperature_offset" and bounds[0] <= 0)):
            raise ValueError(f"Enter valid increasing uncertainty bounds for {key}.")
    operating_draws = np.column_stack([rng.uniform(*conditions[key], samples) for key in keys]) if keys else np.empty((samples, 0))
    draws = np.column_stack((parameter_draws, operating_draws))
    names = [PARAMETERS[index][0] for index in indices] + [dict((item[0], item[1]) for item in CONDITIONS)[key] for key in keys]
    experiment, process = prediction_inputs(case, experiment_index, simulated_profile)
    settings = case.get("pbmsetting", {})
    if settings.get("Numerics") == "jax" and workers > 1:
        raise ValueError("Use one worker with the JAX PBM backend.")
    if progress is not None:
        progress("Simulating the nominal model…")
    nominal = _predict(baseline, experiment, process, case["Solubility"], settings, cancelled=cancelled)
    trial = partial(_trial, baseline=baseline, indices=indices, condition_keys=keys,
                    experiment=experiment, process=process, solubility=case["Solubility"], settings=settings)
    predictions, concentrations, counts, valid, failures = [], [], [], [], []

    def collect(_candidate, result):
        prediction, error = result
        valid.append(prediction is not None)
        failures.append(error or "Success")
        if prediction is not None:
            predictions.append(prediction["percentiles"])
            concentrations.append(prediction["concentration"])
            counts.append(prediction["counts"])

    finished = successful = 0

    def report(result):
        nonlocal finished, successful
        finished += 1
        successful += result[0] is not None
        if draw_progress is not None:
            draw_progress(finished, samples, successful)
        if progress is not None and draw_progress is None:
            progress(f"Simulated {finished}/{samples} draws · {successful} successful · "
                     f"{finished - successful} failed")

    if progress is not None:
        progress(f"Simulating sampled draws with {min(workers, samples)} worker(s)…")
    if draw_progress is not None:
        draw_progress(0, samples, 0)
    if workers > 1:
        with candidate_map(min(workers, samples), trial, collect, cancelled, completed=report) as evaluate:
            evaluate(None, draws)
    else:
        for candidate in draws:
            if cancelled is not None and cancelled():
                raise InterruptedError("Probabilistic analysis stopped.")
            result = trial(candidate, cancelled=cancelled)
            collect(candidate, result)
            report(result)
    if progress is not None:
        progress("Summarizing sampled predictions…")
    if not predictions:
        raise ValueError(f"All sampled simulations failed. First failure. {failures[0]}")
    predictions, concentrations, counts = np.asarray(predictions), np.asarray(concentrations), np.asarray(counts)
    valid = np.asarray(valid, dtype=bool)
    calibration = fbrm_calibration(nominal, experiment)
    count_scale = calibration["scale"] if calibration["scaled"] else 1.
    outputs = np.column_stack((predictions[:, :, -1], concentrations[:, -1], counts[:, -1]*count_scale))
    prcc = partial_rank_correlations(draws[valid], outputs)
    times, observed = observed_percentiles(experiment, nominal["time"][-1])
    request = {"samples": samples, "seed": seed, "use_parameters": use_parameters,
               "conditions": conditions, "workers": workers}
    if simulated_profile:
        request["simulated_profile"] = simulated_profile
    return {"time": nominal["time"], "nominal": nominal, "predictions": predictions,
            "concentrations": concentrations, "counts": counts, "fbrm": calibration,
            "inputs": draws, "input_names": names,
            "valid": valid, "failure_messages": failures, "indices": indices,
            "parameter_mean": baseline[indices], "covariance": covariance, "prcc": prcc,
            "prcc_diagnostics": prcc_diagnostics(draws[valid], outputs),
            "observation_time": dict(zip(("D10", "D50", "D90"), times)),
            "observed": dict(zip(("D10", "D50", "D90"), observed)),
            "model_index": model_index, "experiment_index": experiment_index,
            "request": request}
