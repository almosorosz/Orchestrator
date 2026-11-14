"""Volume-weighted size distributions and percentiles used by IMED."""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, interp1d


def cell_volumes(edges, shape_factor):
    boundary_volumes = shape_factor * edges**3
    return (boundary_volumes[:-1] + boundary_volumes[1:]) / 2


def seed_number_density(experiment, process, edges):
    """Convert the measured volume CSD to the common finite-volume seed state."""
    seed_mass = float(process["SeedMass"])
    if seed_mass == 0:
        return np.zeros(len(edges) - 1)
    if not np.isfinite(seed_mass) or seed_mass < 0:
        raise ValueError("The seed mass must be finite and nonnegative")
    data = np.asarray(experiment["InitialCSDsData"], dtype=float)
    if (data.ndim != 2 or data.shape[0] != 2 or data.shape[1] < 2
            or not np.isfinite(data[0]).all() or np.any(np.diff(data[0]) <= 0)):
        raise ValueError("The seed CSD requires at least two strictly increasing finite sizes")
    frequency = interp1d(*data, fill_value="extrapolate", assume_sorted=True)(edges[1:])
    frequency = np.where(np.isfinite(frequency) & (frequency >= 0), frequency, 0.)
    # Preserve the PBM's lower-edge convention. Extending a measured volume
    # CSD linearly to zero and integrating it over L**3 can give infinite count.
    number_at_edges = np.r_[0., frequency / edges[1:]**3]
    seed = (number_at_edges[:-1] + number_at_edges[1:]) / 2
    volumes = cell_volumes(edges, process["ShapeFactor"])
    represented_mass = np.dot(volumes * np.diff(edges), seed) * process["SolidDensity"] / 1e12
    if not np.isfinite(represented_mass) or represented_mass <= 0:
        raise ValueError("The seed CSD has no finite positive volume on the mesh")
    return seed * (seed_mass / represented_mass)


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
