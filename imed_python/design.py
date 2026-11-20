"""Scoring functions from the IMED model-discrimination and D-optimal objectives."""
from functools import partial
from itertools import combinations, product
from math import prod

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .candidate_workers import candidate_map
from .distribution import cell_volumes, d_values
from .estimation import TRIAL_ERRORS, normalized
from .mesh_grid import grid_edges, is_log_grid
from .solver import model_parameters, simulate


def responses(outputs, process, settings, active_terms, size_term, normalize_count=False):
    values = {}
    for index in active_terms:
        x, y = outputs[index]
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("A simulated response contains negative or nonfinite values")
        if index == 0 and normalize_count:
            y = normalized(y, y[-1])
        if index == 2:
            edges = grid_edges(settings, len(x))
            y = y * cell_volumes(edges, process["ShapeFactor"])
            log_grid = is_log_grid(settings)
            if size_term == "CSD":
                if log_grid:
                    y *= np.diff(edges)
                y = normalized(y, y.sum())
            else:
                y = d_values(x, y, edges=edges if log_grid else None)
        values[index] = y
    return values


def discrimination_score(models, experiment, process, solubility, settings, weights,
                         size_term="CSD", experiment_count=1):
    """Positive weighted pairwise model separation for a proposed experiment."""
    if len(models) < 2 or experiment_count < 1:
        raise ValueError("Model discrimination requires at least two models")
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (3,) or not np.isfinite(weights).all()
            or (weights < 0).any() or not weights.any()):
        raise ValueError("Model discrimination requires nonnegative weights with a positive term")
    active = np.flatnonzero(weights)
    predictions = [
        responses(simulate(model_parameters(model), experiment, process, solubility, settings),
                  process, settings, active, size_term, normalize_count=True)
        for model in models
    ]
    return sum(weights[index] * np.mean((first[index] - second[index])**2)
               for first, second in combinations(predictions, 2)
               for index in active) / experiment_count


def parameter_samples(model, sample_count, seed=0):
    """Latin hypercube sampling of the fitted multivariate normal."""
    active = np.asarray(model["paraset"])[:, 0] == 1
    fitted = np.asarray(model.get("para", []), dtype=float).reshape(-1)
    covariance = np.atleast_2d(np.asarray(model.get("covmat", []), dtype=float))
    size = int(active.sum())
    if (sample_count < 1 or not size or fitted.size != size or covariance.shape != (size, size)
            or not np.isfinite(fitted).all() or not np.isfinite(covariance).all()):
        raise ValueError("D-optimal design requires a fitted model with finite covariance")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("Parameter covariance must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = np.finfo(float).eps * size * np.max(np.abs(eigenvalues))
    if (eigenvalues < -tolerance).any():
        raise ValueError("Parameter covariance must be positive semidefinite")
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    unit = qmc.LatinHypercube(size, seed=seed).random(sample_count)
    return fitted + ndtri(unit) @ factor.T


def d_optimal_log_score(model, experiment, process, solubility, settings, enabled_terms, samples, *, size_term="CSD"):
    """Log determinant of the Fisher matrix using IMED's relative sensitivities."""
    parameters = model_parameters(model)
    estimated = np.flatnonzero(np.asarray(model["paraset"])[:, 0] == 1)
    active = np.flatnonzero(enabled_terms)
    samples = np.asarray(samples, dtype=float)
    if (len(enabled_terms) != 3 or not len(active) or not len(estimated)
            or samples.ndim != 2 or samples.shape[1] != len(estimated)
            or not len(samples) or not np.isfinite(samples).all()):
        raise ValueError("D-optimal design requires an enabled response and valid parameter samples")

    def evaluate(values):
        outputs = simulate(values, experiment, process, solubility, settings)
        return responses(outputs, process, settings, active, size_term)

    base = evaluate(parameters)
    columns = []
    for index, values in zip(estimated, samples.T):
        nominal = parameters[index]
        distinct = values[values != nominal]
        if not len(distinct):
            return -np.inf
        if nominal == 0:
            return -np.inf
        perturbed = parameters.copy()
        sampled = {term: [] for term in active}
        for value in distinct:
            perturbed[index] = value
            for term, output in evaluate(perturbed).items():
                sampled[term].append(output)
        column = []
        for term, outputs in sampled.items():
            outputs = np.stack(outputs)
            relative_delta = ((distinct - nominal) / nominal)[:, None]
            differences = np.abs((outputs - base[term]) / relative_delta).mean(axis=0)
            span = np.ptp(outputs)
            column.append(differences / span if span > 0 else np.zeros_like(differences))
        columns.append(np.concatenate(column))
    sensitivity = np.column_stack(columns)
    if np.linalg.matrix_rank(sensitivity) < len(estimated):
        return -np.inf
    sign, log_determinant = np.linalg.slogdet(sensitivity.T @ sensitivity)
    return float(log_determinant) if sign > 0 else -np.inf


def _candidate_recipe(candidate, *, experiment, process, conditions, addition):
    batch = conditions["batch"]
    start_t, end_t = conditions["temperature"]
    solvent, antisolvent = conditions["liquid_masses"]
    concentration = conditions["concentration"]
    seed_mass, ramp_duration, ramp_position, pulse_duration, pulse_position = candidate
    if end_t != start_t:
        ramp_start = ramp_position * (batch - ramp_duration)
        t = np.array([0, ramp_start, ramp_start + ramp_duration, batch])
        keep = np.r_[True, np.diff(t) > 0]
        temperature = np.vstack((t[keep], np.array([start_t, start_t, end_t, end_t])[keep]))
    else:
        temperature = np.array([[0, batch], [start_t, start_t]])
    if addition:
        pulse_start = pulse_position * (batch - pulse_duration - 0.1)
        t = np.array([0, pulse_start, pulse_start + 0.1, pulse_start + pulse_duration,
                      pulse_start + pulse_duration + 0.1, batch])
        keep = np.r_[True, np.diff(t) > 0]
        rate = addition / pulse_duration
        flow = np.vstack((t[keep], np.array([0, 0, rate, rate, 0, 0])[keep]))
    else:
        flow = np.array([[0, batch], [0, 0]])
    equilibration = conditions.get("equilibration_s", 0.)
    if equilibration:
        temperature = np.column_stack((temperature, [batch + equilibration, end_t]))
        flow = np.column_stack((flow, [batch + equilibration, 0.]))
    proposed = {**experiment, "TemperatureData": temperature, "FlowRateData": flow,
                "ConcentrationData": np.array([[0], [concentration]]),
                "InitialConcentration": concentration}
    design_process = {**process, "SeedMass": seed_mass, "SolvMass": solvent,
                      "AntiSolvMass": antisolvent}
    if "InitialVolume_ml" in process:
        carrier = process["SolvMass"] + process["AntiSolvMass"]
        same_composition = carrier > 0 and np.isclose(process["AntiSolvMass"] / carrier, antisolvent / (solvent + antisolvent))
        initial_volume = (process["InitialVolume_ml"] * (solvent + antisolvent) / carrier if same_composition
                          else solvent / process["SolventDensity"] + antisolvent / process["AntiSolventDensity"])
        design_process.update(InitialVolume_ml=initial_volume, RequestedInitialVolume_ml=initial_volume, VolumeFromDesignConditions=True)
    proposed.pop("AddedAntisolventData", None)
    if "MixingSettings" in process:
        from .mixing import average_mixing
        design_process = average_mixing(proposed, design_process, process["MixingSettings"], mean_rpm=process.get("StirringRPM"))
    return proposed, design_process


def _recipe_cost(score, proposed, process):
    try:
        value = score(proposed, process)
    except TRIAL_ERRORS:
        return np.inf
    return -value if np.isfinite(value) else np.inf


def _design_trial(candidate, *, recipe, score, cancelled):
    if cancelled():
        raise InterruptedError("Experiment design cancelled")
    try:
        return _recipe_cost(score, *recipe(candidate))
    except TRIAL_ERRORS:
        return np.inf


def recipe_space(conditions):
    batch = conditions["batch"]
    start_t, end_t = conditions["temperature"]
    solvent, antisolvent = conditions["liquid_masses"]
    concentration = conditions["concentration"]
    final_anti = conditions["final_anti"]
    seed_bounds = conditions["seed_bounds"]
    if (batch <= 0 or solvent <= 0 or antisolvent < 0 or not 0 <= concentration < 1
            or not 0 <= final_anti < 1 or not 0 <= seed_bounds[0] <= seed_bounds[1]):
        raise ValueError("Check batch time, starting masses, concentration, and seed bounds")
    add = 0. if conditions.get("no_feed") else final_anti / (1 - final_anti) * solvent - antisolvent
    if add < -1e-10:
        raise ValueError("Final antisolvent fraction is below the initial fraction")
    add = max(0.0, add)
    delta_t = abs(end_t - start_t)

    def duration_bounds(change, rates, available, minimum):
        if not change:
            return 0., 0.
        low_rate, high_rate = rates
        if not 0 <= low_rate <= high_rate or high_rate <= 0:
            raise ValueError("Set ordered rate bounds with a positive maximum")
        low = max(minimum, change / high_rate)
        high = min(available, change / low_rate) if low_rate else available
        if low > high:
            raise ValueError("No feasible duration fits the selected batch time and rate bounds")
        return low, high

    ramp = duration_bounds(delta_t, conditions["temperature_rates"], batch, 0)
    pulse = duration_bounds(add, conditions["antisolvent_rates"], batch - 0.1, 0.1)
    return add, [seed_bounds, ramp, (0, int(bool(delta_t))), pulse, (0, int(bool(add)))]


def optimize_recipe(experiment, process, conditions, score, *, iterations=5, population=12,
                    cancelled=None, workers=1, retain_partial=False):
    """Optimize a recipe. Parallel searches require a picklable score callable."""
    from scipy.optimize import differential_evolution

    integrality = None
    if (doe := conditions.get("doe_search")) is not None:
        from .doe_design import DoeRecipe

        controls = doe["controls"]
        recipe = DoeRecipe(experiment, process, doe["space"], controls,
                           seed_template=doe.get("seed_template"), final_hold_s=doe.get("final_hold_s", 0.)).candidate
        bounds = [(low, high) for low, high, _ in controls.values()]
        integrality = [bool(discrete) for _, _, discrete in controls.values()]
    else:
        add, bounds = recipe_space(conditions)
        recipe = partial(_candidate_recipe, experiment=experiment, process=process,
                         conditions=conditions, addition=add)
    best = None

    def evaluate(candidate):
        nonlocal best
        if cancelled is not None and cancelled():
            raise InterruptedError("Experiment design cancelled")
        try:
            proposed, design_process = recipe(candidate)
        except TRIAL_ERRORS:
            return np.inf
        cost = _recipe_cost(score, proposed, design_process)
        if np.isfinite(cost) and (best is None or -cost > best[0]):
            best = -cost, proposed, design_process
        return cost

    middle = np.mean(bounds, axis=1) if bounds else np.empty(0)
    if integrality is not None and bounds:
        middle[integrality] = np.rint(middle[integrality])
    free = sum(high > low for low, high in bounds)
    choices = None
    if integrality and all(integrality) and workers == 1:
        count = prod(int(high - low) + 1 for low, high in bounds)
        if count <= max(population, min(128, population * (iterations + 1))):
            choices = product(*(range(int(low), int(high) + 1) for low, high in bounds))
    try:
        if not free:
            evaluate(middle)
        elif choices is not None:
            for candidate in choices:
                evaluate(candidate)
        else:
            popsize = max(5, int(np.ceil(population / free)))
            options = dict(x0=middle, maxiter=iterations, popsize=popsize, polish=False, seed=0,
                           integrality=integrality)
            if workers == 1:
                differential_evolution(evaluate, bounds, **options)
            else:
                trial = partial(_design_trial, recipe=recipe, score=score)

                def collect(candidate, cost):
                    nonlocal best
                    if np.isfinite(cost) and (best is None or -cost > best[0]):
                        best = -cost, *recipe(candidate)
                    return cost

                with candidate_map(min(workers, popsize * free), trial, lambda _, cost: cost,
                                   cancelled, retain=collect) as map_trials:
                    differential_evolution(evaluate, bounds, workers=map_trials,
                                           updating="deferred", **options)
    except InterruptedError:
        if not retain_partial or best is None:
            raise
    if best is None:
        raise ValueError("No candidate produced a finite objective. Check the model, data, and rate bounds.")
    return best
