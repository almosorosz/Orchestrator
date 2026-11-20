"""Monotone fits for pump calibration data."""
import numpy as np
from scipy.optimize import minimize_scalar


def monotone_quadratic(x, y):
    """Fit y = b*x + c*x² with nonnegative slope over the observed range."""
    xmax = max(x)
    x, y = np.asarray(x) / xmax, np.asarray(y)
    design = np.column_stack((x, x**2))
    if len(np.unique(x[x > 0])) == 1:
        b, c = x @ y / (x @ x), 0.
    else:
        b, c = np.linalg.lstsq(design, y, rcond=None)[0]
        if b < 0 or b + 2*c < 0:
            # The constrained optimum lies on one of the two slope boundaries.
            end_flat = x - .5*x**2
            slope = max(0., end_flat @ y / (end_flat @ end_flat))
            pure_quadratic = max(0., x**2 @ y / (x**2 @ x**2))
            b, c = min(((slope, -.5*slope), (0., pure_quadratic)),
                       key=lambda pair: np.linalg.norm(design @ pair - y))
    return [float(b / xmax), float(c / xmax**2)]


def fit_seed_exponential(points):
    """Fit dry mass after the last elapsed time with no delivery."""
    time, mass = points.T
    delay = float(time[mass == 0][-1])
    span = time[-1] - delay
    x = np.maximum(time - delay, 0.) / span

    def fit(exponent):
        basis = np.expm1(exponent * x) / exponent if exponent else x
        scale = (basis @ mass) / (basis @ basis)
        return np.linalg.norm(scale*basis - mass), scale

    # Fit the curvature separately from the linear scale. The zero exponent is
    # the constant-delivery limit and also covers a log with only one dry dose.
    exponent = 0.
    if (mass > 0).sum() > 1:
        result = minimize_scalar(lambda value: fit(value)[0], bounds=(0., 32.), method="bounded")
        if result.success and fit(result.x)[0] < fit(0.)[0] - 1e-10 * np.linalg.norm(mass):
            exponent = result.x
    return {"delay_s": delay, "initial_rate_g_s": float(fit(exponent)[1] / span), "k_s_inv": float(exponent / span)}


def seed_exponential_mass(model, seconds):
    elapsed = np.maximum(np.asarray(seconds) - model["delay_s"], 0.)
    rate, k = model["initial_rate_g_s"], model["k_s_inv"]
    with np.errstate(over="ignore", invalid="ignore"):
        mass = (rate / k) * np.expm1(k * elapsed) if k else rate * elapsed
    if not np.isfinite(mass).all():
        raise ValueError("The seed pulse is too long to evaluate the exponential fit.")
    return mass


def seed_exponential_time(model, grams):
    rate, k = model["initial_rate_g_s"], model["k_s_inv"]
    seconds = np.log1p(grams * (k / rate)) / k if k else grams / rate
    if not np.isfinite(seconds):
        raise ValueError("The seed dose is too large to evaluate the exponential fit.")
    return float(model["delay_s"] + seconds)
