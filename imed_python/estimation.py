"""Residuals and two-stage SciPy fitting for IMED."""
from copy import copy
from dataclasses import dataclass
from functools import partial
from time import monotonic

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

from .candidate_workers import candidate_map
from .distribution import cell_volumes, d_values
from .solver import pbm_backend_label, simulate
from .size_data import TREND_SIZE_TERM
from .uncertainty import DISCRETE_PARAMETERS, ParameterIntervals, intervals_from_covariance


PYFIT_SETTINGS_KEY = "pyfitsettings"
PYFIT_DEFAULTS = {
    "global_iterations": 20,
    "population_size": 200,
    "global_tolerance": 0.01,
    "mutation_min": 0.5,
    "mutation_max": 1.0,
    "recombination": 0.7,
    "local_evaluations": 1500,
}
TRIAL_ERRORS = (ValueError, RuntimeError, TimeoutError, FloatingPointError)
REJECTED_RESIDUAL = 1e4


def series(value, name):
    points = np.asarray(value, dtype=float)
    if points.shape == (2,):
        points = points[:, None]
    if (points.ndim != 2 or points.shape[0] != 2 or not points.shape[1]
            or not np.isfinite(points).all()
            or np.any(np.diff(points[0]) <= 0)):
        raise ValueError(f"{name} requires finite data with increasing coordinates")
    return points


def normalized(values, denominator):
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Cannot normalize data with a nonpositive denominator")
    return values / denominator


class FitObjective:
    def __init__(self, experiments, processes, solubility, settings, weights, terms, *,
                 cancelled=None, at_measurements=False):
        if not len(experiments) or len(experiments) != len(processes):
            raise ValueError("Each experiment requires a process information record")
        self.settings = settings
        self.solubility = solubility
        self.terms = terms
        self.cancelled = cancelled
        self.at_measurements = at_measurements
        self.weights = np.asarray(weights, dtype=float)
        if (self.weights.shape != (3,) or not np.isfinite(self.weights).all()
                or (self.weights < 0).any() or not self.weights.any()):
            raise ValueError("Set nonnegative weights with at least one positive term")
        self.experiments = []
        self.residual_count = 0
        self.sample_count = 0
        edges = np.linspace(settings.get("Lmin", 0), settings.get("Lmax", 250),
                            int(settings.get("N", 250)) + 1)
        centers = edges[1:] - np.diff(edges) / 2
        if at_measurements and centers.size < 2:
            raise ValueError("Use a PBM grid with at least two bins.")
        for experiment, process in zip(experiments, processes):
            observed = {}
            for index, key in enumerate(("CountsData", "ConcentrationData", "FinalCSDsData")):
                if not self.weights[index]:
                    continue
                if index == 2 and terms == TREND_SIZE_TERM:
                    end = np.asarray(experiment["TemperatureData"], dtype=float).reshape(2, -1)[0].max()
                    final = np.asarray(experiment.get("FinalD10D50D90", []), dtype=float).reshape(-1)
                    traces = {}
                    for column, name in enumerate(("D10Data", "D50Data", "D90Data")):
                        points = np.empty((2, 0))
                        if np.size(experiment.get(name, [])):
                            points = series(experiment[name], name)
                            points = points[:, (points[0] >= 0) & (points[0] <= end) & (points[1] > 0)]
                        if not points.size and final.size == 3 and np.isfinite(final[column]) and final[column] > 0:
                            points = np.array([[end], [final[column]]])
                        if points.size:
                            traces[column] = points
                    if not traces:
                        if at_measurements:
                            continue
                        raise ValueError("Trend fitting requires recorded D10/D50/D90 data or final measured sizes.")
                    observed[index] = traces
                    count = sum(points.shape[1] for points in traces.values())
                    self.residual_count += count
                    self.sample_count += count
                    continue
                percentiles = index == 2 and terms != "CSD"
                field = "FinalD10D50D90" if percentiles else key
                data = experiment.get(field)
                if at_measurements and (data is None or not np.size(data)):
                    continue
                if percentiles:
                    values = np.asarray(experiment[field], dtype=float).reshape(-1)
                    if values.shape != (3,) or not np.isfinite(values).all():
                        raise ValueError("FinalD10D50D90 requires three finite values")
                    observed[index] = values
                    self.residual_count += 3
                    self.sample_count += 3
                else:
                    points = series(experiment[field], field)
                    if at_measurements:
                        if index == 1:
                            points = points[:, points[0] > 0]
                        elif index == 2:
                            points = points[:, (points[0] >= centers[0]) & (points[0] <= centers[-1])]
                        if points.shape[1] <= int(index != 1):
                            continue
                    observed[index] = points
                    self.residual_count += (len(centers) if index == 2 and not at_measurements
                                            else points.shape[1])
                    # Peak or total normalization removes one independent component.
                    self.sample_count += points.shape[1] - int(at_measurements and index != 1)
            if at_measurements and not observed:
                continue
            volumes = cell_volumes(edges, process["ShapeFactor"])
            self.experiments.append((experiment, process, observed, volumes))
        if not self.experiments:
            raise ValueError("Select a response with usable measurements inside the PBM grid.")

    def residuals(self, parameters, *, global_stage=False):
        self.simulation_times = []
        result = []
        for experiment_number, (experiment, process, observed, volumes) in enumerate(self.experiments, 1):
            trends = self.terms == TREND_SIZE_TERM and 2 in observed
            started = monotonic()
            try:
                simulation = simulate(parameters, experiment, process, self.solubility, self.settings,
                                      return_history=trends, cancelled=self.cancelled)
                outputs = simulation[:3]
            except (FloatingPointError, TimeoutError) as error:
                raise type(error)(f"Experiment {experiment_number}: {error}") from error
            finally:
                self.simulation_times.append(monotonic() - started)
            for index, measured in observed.items():
                x, prediction = outputs[index]
                label = ("particle count", "concentration", "size distribution")[index]
                if not np.isfinite(prediction).all():
                    raise ValueError(f"Experiment {experiment_number} has nonfinite predicted {label}.")
                if np.any(prediction < 0):
                    worst = np.argmin(prediction)
                    unit = "µm" if index == 2 else "s"
                    raise ValueError(f"Experiment {experiment_number} has negative predicted {label} "
                                     f"({prediction[worst]:.5g}) at {x[worst]:.6g} {unit}.")
                if index == 2:
                    volume_density = prediction * volumes
                    if trends:
                        from .comparison import percentile_history

                        percentiles = percentile_history(outputs[2], simulation[3], process, self.settings,
                                                          cancelled=self.cancelled)
                        difference = np.concatenate([
                            np.interp(points[0], outputs[0][0], percentiles[column], left=np.nan, right=np.nan)
                            - points[1] for column, points in measured.items()])
                        count = len(difference)
                    elif self.terms == "CSD":
                        if self.at_measurements:
                            predicted = np.interp(measured[0], x, volume_density,
                                                  left=np.nan, right=np.nan)
                            observed_density = measured[1]
                            count = len(predicted)
                        else:
                            predicted = volume_density
                            observed_density = np.interp(x, *measured, left=np.nan, right=np.nan)
                            count = len(x) if global_stage else measured.shape[1]
                        difference = (normalized(predicted, predicted.sum())
                                      - normalized(observed_density, observed_density.sum()))
                    else:
                        difference = d_values(x, volume_density) - measured
                        count = 3
                else:
                    predicted = np.interp(measured[0], x, prediction, left=np.nan, right=np.nan)
                    values = measured[1]
                    if index == 0:
                        # The original scalar and vector objectives use different count scaling.
                        reference = (lambda data: data[-1]) if global_stage else np.max
                        predicted = normalized(predicted, reference(predicted))
                        values = normalized(values, reference(values))
                    difference = predicted - values
                    count = len(values)
                if not np.isfinite(difference).all():
                    raise ValueError("Experimental observations fall outside the simulation grid")
                scale = np.sqrt(self.weights[index] / (len(self.experiments) * count))
                result.append(scale * difference)
        return np.concatenate(result)


@dataclass
class FitResult:
    parameters: np.ndarray
    residuals: np.ndarray
    jacobian: np.ndarray
    covariance: np.ndarray
    ssr: float
    aic: float
    converged: bool
    intervals: ParameterIntervals | None = None
    uncertainty_error: str = ""
    simulation_count: int = 0
    simulation_seconds: float = 0.
    wall_seconds: float = 0.


def _global_trial(candidate, *, objective, baseline, free, cancelled):
    if cancelled():
        raise InterruptedError("Parameter estimation cancelled")
    objective.cancelled = cancelled
    parameters = baseline.copy()
    parameters[free] = candidate
    try:
        errors = objective.residuals(parameters, global_stage=True)
    except TRIAL_ERRORS as error:
        return objective.residual_count * REJECTED_RESIDUAL**2, str(error), getattr(objective, "simulation_times", [])
    return float(errors @ errors), None, getattr(objective, "simulation_times", [])


def fit_model(model, objective, *, global_iterations=PYFIT_DEFAULTS["global_iterations"],
              population_size=PYFIT_DEFAULTS["population_size"],
              global_tolerance=PYFIT_DEFAULTS["global_tolerance"],
              mutation_min=PYFIT_DEFAULTS["mutation_min"], mutation_max=PYFIT_DEFAULTS["mutation_max"],
              recombination=PYFIT_DEFAULTS["recombination"],
              local_evaluations=PYFIT_DEFAULTS["local_evaluations"], cancelled=None, progress=None,
              workers=1):
    """Fit the model, optionally evaluating global trials in separate CPU processes."""
    if isinstance(objective, FitObjective) and objective.settings.get("Numerics") == "jax":
        if workers != 1:
            raise ValueError("Use one worker for JAX PBM.")
        pbm_backend_label(objective.settings)

    definitions = np.asarray(model["paraset"], dtype=float)
    active = definitions[:, 0] == 1
    if not active.any():
        raise ValueError("Enable estimation for at least one parameter")
    lower, upper = definitions[active, 2:].T
    if not np.isfinite([lower, upper]).all() or np.any(upper < lower):
        raise ValueError("Estimated parameters require finite ordered bounds")
    free = upper > lower
    parameters = definitions[:, 1].copy()
    fitted = np.asarray(model.get("para", []), dtype=float).reshape(-1)
    initial = fitted if fitted.size == active.sum() and np.isfinite(fitted).all() else parameters[active]
    values = np.clip(initial, lower, upper)

    phase = ""
    evaluations = rejected = generation = 0
    best = None
    current = None
    next_update = 0.
    last_report = None
    latest_times = []
    total_simulations = 0
    simulation_seconds = 0.
    fit_started = monotonic()

    def begin_phase(name):
        nonlocal phase, evaluations, rejected, generation, best, current, next_update, last_report
        phase = name
        evaluations = rejected = generation = 0
        best = current = None
        next_update = monotonic() + 1.
        last_report = None
        if progress is not None:
            progress(name)

    def report_progress(*, force=False):
        nonlocal next_update, last_report
        if progress is None or (not force and monotonic() < next_update):
            return
        next_update = monotonic() + 1.
        unit = "evaluation" if evaluations == 1 else "evaluations"
        line = f"{phase} · {evaluations:,} trial {unit} · {rejected:,} rejected"
        if evaluations:
            value = "trial rejected" if current is None else f"{current:.5g}"
            line += f" · current objective {value}"
        if generation:
            line += f" · generation {generation}/{int(global_iterations)} · best WMSE {best:.5g}"
        if latest_times:
            line += (f" · PBM {len(latest_times)} simulation(s) in {sum(latest_times):.3f} s"
                     f" · mean {np.mean(latest_times):.3f} s")
        if line != last_report:
            progress(line)
            last_report = line

    def report_generation(intermediate_result):
        nonlocal generation, best
        generation = intermediate_result.nit
        best = intermediate_result.fun
        report_progress(force=generation == 1)

    def record_timing(times):
        nonlocal latest_times, total_simulations, simulation_seconds
        latest_times = times
        total_simulations += len(times)
        simulation_seconds += sum(times)

    def record_trial(score, error, times=None):
        nonlocal evaluations, rejected, current
        record_timing(getattr(objective, "simulation_times", []) if times is None else times)
        if error is not None:
            rejected += 1
            if rejected == 1 and progress is not None:
                progress(f"{phase} · first rejected trial: {error}")
        current = score if error is None else None
        evaluations += 1
        report_progress(force=evaluations == 1)
        return score

    def residual(candidate, global_stage=False, penalize=True):
        if cancelled is not None and cancelled():
            raise InterruptedError("Parameter estimation cancelled")
        values[free] = candidate
        parameters[active] = values
        try:
            errors = objective.residuals(parameters, global_stage=global_stage)
        except TRIAL_ERRORS as error:
            if not penalize:
                raise
            errors = np.full(objective.residual_count, REJECTED_RESIDUAL)
            record_trial(None, str(error))
        else:
            if penalize:
                record_trial(float(errors @ errors) if progress is not None else None, None)
            else:
                record_timing(getattr(objective, "simulation_times", []))
        return errors

    def score(candidate):
        errors = residual(candidate, global_stage=True)
        return float(errors @ errors)

    start = values[free]
    if free.any() and global_iterations:
        begin_phase("Differential evolution" if workers == 1 else
                    f"Differential evolution ({workers} CPU workers)")
        bounds = list(zip(lower[free], upper[free]))
        # SciPy's popsize is a multiplier of the number of free parameters.
        # Supplying the initial sample preserves IMED's absolute population size.
        unit_population = qmc.LatinHypercube(d=len(bounds), seed=0).random(n=max(5, int(population_size)))
        population = qmc.scale(unit_population, lower[free], upper[free])
        options = dict(maxiter=int(global_iterations), seed=0, x0=start, init=population,
                       tol=float(global_tolerance), polish=False,
                       mutation=(float(mutation_min), float(mutation_max)),
                       recombination=float(recombination),
                       callback=report_generation if progress else None)
        if workers == 1:
            global_result = differential_evolution(score, bounds, **options)
        else:
            worker_objective = copy(objective)
            worker_objective.cancelled = None
            baseline = parameters.copy()
            baseline[active] = values
            free_parameters = np.flatnonzero(active)[free]
            trial = partial(_global_trial, objective=worker_objective, baseline=baseline,
                            free=free_parameters)

            def collect(_candidate, result):
                return record_trial(*result)

            objective_stop = getattr(objective, "cancelled", None)

            def stop_parallel():
                return ((cancelled is not None and cancelled())
                        or (objective_stop is not None and objective_stop is not cancelled and objective_stop()))

            with candidate_map(min(workers, len(population)), trial, collect, stop_parallel) as map_trials:
                global_result = differential_evolution(score, bounds, workers=map_trials,
                                                       updating="deferred", **options)
        start = global_result.x
        report_progress(force=True)
    jacobian = np.zeros((objective.residual_count, int(active.sum())))
    converged = True
    if free.any():
        begin_phase("Local least-squares refinement")
        result = least_squares(residual, start, bounds=(lower[free], upper[free]),
                               max_nfev=int(local_evaluations), ftol=1e-5)
        start = result.x
        jacobian[:, free] = result.jac
        converged = result.success
        report_progress(force=True)
    if progress is not None:
        progress("Checking the final fit")
    errors = residual(start, penalize=False)
    global_errors = residual(start, global_stage=True, penalize=False)
    ssr = float(global_errors @ global_errors)
    degrees = len(errors) - int(free.sum())
    covariance = np.full((len(values), len(values)), np.nan)
    if degrees > 0 and (not free.any() or np.linalg.matrix_rank(jacobian[:, free]) == free.sum()):
        covariance = (errors @ errors / degrees) * np.linalg.pinv(jacobian.T @ jacobian)
    sample_count = objective.sample_count
    nvars = len(values)
    aic = np.nan
    if sample_count > nvars + 2 and ssr > 0:
        aic = np.log(ssr) + (sample_count + nvars) / (sample_count - nvars - 2)
    intervals = None
    uncertainty_error = ""
    selected = np.flatnonzero(active)
    positions = np.flatnonzero(free & ~np.isin(selected, DISCRETE_PARAMETERS))
    if progress is not None:
        progress("Calculating parameter confidence intervals")
    try:
        if not positions.size:
            raise ValueError("No freely estimated continuous kinetic parameters.")
        intervals = intervals_from_covariance(
            selected[positions], values[positions], covariance[np.ix_(positions, positions)], degrees)
    except ValueError as error:
        uncertainty_error = str(error)
        if progress is not None:
            progress(f"Confidence intervals unavailable · {error}")
    wall_seconds = monotonic() - fit_started
    if progress is not None:
        average = simulation_seconds / total_simulations if total_simulations else 0.
        progress(f"Simulation timing · {total_simulations} PBM runs · total compute {simulation_seconds:.3f} s"
                 f" · mean {average:.3f} s/run · fit wall time {wall_seconds:.3f} s")
    return FitResult(values.copy(), errors, jacobian, covariance, ssr, aic, converged,
                     intervals, uncertainty_error, total_simulations, simulation_seconds, wall_seconds)
