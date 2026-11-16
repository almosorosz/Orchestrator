"""Volume-weighted size distributions and percentiles used by IMED."""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline


def cell_volumes(edges, shape_factor):
    boundary_volumes = shape_factor * edges**3
    return (boundary_volumes[:-1] + boundary_volumes[1:]) / 2


def volume_distribution(csd, process, settings):
    length, number_density = csd
    edges = np.linspace(settings.get("Lmin", 0), settings.get("Lmax", 250), len(length) + 1)
    density = np.maximum(number_density, 0) * cell_volumes(edges, process["ShapeFactor"])
    distribution = percent_density(length, density)
    if distribution is None:
        return np.vstack((length, np.zeros_like(length))), None
    return distribution, d_values(length, density)


def percent_density(length, density):
    integral = np.trapezoid(density, length)
    if integral <= 0 or not np.isfinite(integral):
        return None
    return np.vstack((length, 100 * density / integral))


def d_values(length, density):
    x = np.linspace(length[0], length[-1], 100_000)
    pdf = np.maximum(np.interp(x, length, density), 0)
    cdf = cumulative_trapezoid(pdf, x, initial=0)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("The predicted size distribution has no particle volume")
    cdf, indices = np.unique(cdf / cdf[-1], return_index=True)
    return CubicSpline(cdf, x[indices])([0.1, 0.5, 0.9])
