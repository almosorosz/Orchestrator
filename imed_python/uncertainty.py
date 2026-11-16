"""Local confidence intervals from PBM residuals at the current model values."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t

from .parameter_labels import PARAMETERS
from .solver import model_parameters


# The slurry dependency, agglomeration kernel, and daughter distribution are switches.
DISCRETE_PARAMETERS = (8, 26, 34)
SENSITIVITY_ERRORS = (ValueError, RuntimeError, TimeoutError, FloatingPointError)


@dataclass(frozen=True)
class ParameterIntervals:
    indices: np.ndarray
    values: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    degrees_of_freedom: int
    source: str
    convention: str = "matlab"


def intervals_from_covariance(indices, values, covariance, degrees, *, confidence=0.95,
                              source="Fitted values", convention="matlab"):
    """Use MATLAB IMED's normal multiplier or a finite-sample t multiplier."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0% and 100%.")
    if convention not in ("matlab", "student_t"):
        raise ValueError("Choose the MATLAB IMED or Student t CI convention.")
    covariance = np.asarray(covariance, dtype=float)
    if (degrees <= 0 or covariance.shape != (len(values), len(values))
            or not np.isfinite(covariance).all() or np.any(np.diag(covariance) < 0)):
        raise ValueError("The covariance is unavailable or there are insufficient residual degrees of freedom.")
    standard_errors = np.sqrt(np.diag(covariance))
    distribution = norm if convention == "matlab" else t(degrees)
    half_width = distribution.isf((1 - confidence) / 2) * standard_errors
    if not np.isfinite(half_width).all():
        raise ValueError("The local intervals are too large to calculate reliably.")
    values = np.asarray(values, dtype=float)
    return ParameterIntervals(np.asarray(indices), values, covariance, standard_errors,
                              values - half_width, values + half_width, confidence, degrees,
                              source, convention)


def sensitivity_column(index, parameters, baseline, lower, upper, residuals, *, retry=None):
    """Use a feasible one-sided derivative if one trial fails, and shrink if both fail."""
    value = parameters[index]
    step = 1e-3 * max(1., abs(value))
    minimum = 8 * np.spacing(max(abs(value), np.finfo(float).tiny))
    failures = {}
    trial = parameters.copy()
    for attempt in range(4):
        # Prefer symmetric steps away from a bound, and prefer the inward direction at a bound.
        low, high = max(lower, value - step), min(upper, value + step)
        offsets = sorted((point for point in (low, high)
                          if point != value and abs(point - value) >= minimum),
                         key=lambda point: abs(point - value), reverse=True)
        usable = {}
        centered = min(value - lower, upper - value) >= step
        for point in offsets:
            if usable and not centered:
                break
            if point in failures:
                continue
            trial[index] = point
            try:
                responses = residuals(trial)
                if not np.isfinite(responses).all():
                    raise FloatingPointError("nonfinite residuals")
                usable[point] = responses
            except SENSITIVITY_ERRORS as error:
                failures[point] = error

        if len(usable) == 2:
            low, high = sorted(usable)
            return (usable[high] - usable[low]) / (high - low)
        if usable:
            point, responses = next(iter(usable.items()))
            return (responses - baseline) / (point - value)
        if not offsets:
            break
        step *= 0.1
        if retry is not None and attempt < 3:
            retry()

    name = PARAMETERS[index][0]
    if failures:
        point, error = next(reversed(failures.items()))
        raise ValueError(f"Could not calculate sensitivity for {name} near {value:.6g}. "
                         f"Last failed trial at {point:.8g}: {error}") from error
    raise ValueError(f"No representable sensitivity step inside the bounds for {name}.")


def local_intervals(model, objective, *, confidence=0.95, progress=None, convention="matlab"):
    """Linearize measured residuals without optimizing or modifying the model."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0% and 100%.")
    definitions = np.asarray(model["paraset"], dtype=float)
    if definitions.shape != (len(PARAMETERS), 4):
        raise ValueError("The model needs a valid kinetic parameter table.")
    active = definitions[:, 0] == 1
    continuous = active.copy()
    continuous[list(DISCRETE_PARAMETERS)] = False
    bounds = definitions[:, 2:4]
    if not np.isfinite(bounds[continuous]).all() or np.any(bounds[continuous, 1] < bounds[continuous, 0]):
        raise ValueError("Selected parameters require finite, ordered bounds.")
    indices = np.flatnonzero(continuous & (bounds[:, 1] > bounds[:, 0]))
    if not indices.size:
        raise ValueError("In Model Configurations, check Estimate and give at least one continuous "
                         "kinetic parameter distinct lower and upper bounds.")
    fitted = np.asarray(model.get("para", []), dtype=float).reshape(-1)
    if fitted.size and (fitted.size != active.sum() or not np.isfinite(fitted).all()):
        raise ValueError("The stored fit does not match the selected parameters.")
    parameters = model_parameters(model)
    values = parameters[indices]
    lower, upper = bounds[indices].T
    if not np.isfinite(values).all() or np.any(values < lower) or np.any(values > upper):
        raise ValueError("Current parameter values must be finite and within their configured bounds.")
    observations = (getattr(objective, "residual_count", objective.sample_count)
                    if convention == "matlab" else objective.sample_count)
    degrees = observations - len(indices)
    if degrees <= 0:
        raise ValueError("More independent measurements than selected parameters are needed. "
                         "Add measured responses or select fewer parameters.")

    def residuals(point):
        if objective.cancelled is not None and objective.cancelled():
            raise InterruptedError("Uncertainty calculation stopped.")
        return objective.residuals(point)

    try:
        errors = residuals(parameters)
    except SENSITIVITY_ERRORS as error:
        raise ValueError(f"At the current parameter values, {error}") from error
    variance = float(errors @ errors) / degrees
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("The residuals do not provide a positive, finite error variance.")
    jacobian = np.empty((len(errors), len(indices)))
    for column, index in enumerate(indices):
        label = f"{PARAMETERS[index][0]} ({column + 1}/{len(indices)})"
        if progress is not None:
            progress(f"Checking sensitivity for {label}…")
        jacobian[:, column] = sensitivity_column(
            index, parameters, errors, lower[column], upper[column], residuals,
            retry=(lambda: progress(f"Checking sensitivity for {label} · reducing the step…"))
            if progress is not None else None)

    if progress is not None:
        progress("Sensitivity evaluations complete. Calculating covariance and confidence intervals…")
    scales = np.linalg.norm(jacobian, axis=0)
    message = ("The measurements cannot distinguish all selected parameters. "
               "Select fewer parameters or add informative measured responses.")
    if not np.isfinite(scales).all() or np.any(scales == 0):
        raise ValueError(message)
    _, singular, directions = np.linalg.svd(jacobian / scales, full_matrices=False)
    # Finite differences are less precise than an analytic Jacobian.
    threshold = singular[0] * np.sqrt(np.finfo(float).eps)
    if len(singular) < len(indices) or singular[-1] <= threshold:
        raise ValueError(message)
    factor = (directions.T / scales[:, None]) / singular
    covariance = variance * (factor @ factor.T)
    return intervals_from_covariance(
        indices, values, covariance, degrees, confidence=confidence, convention=convention,
        source="Fitted values" if fitted.size else "Nominal values")
