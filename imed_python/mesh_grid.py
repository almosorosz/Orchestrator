"""Physical particle-size bin edges shared by the PBM and its analysis tools."""

import numpy as np


DEFAULT_LOG_OFFSET = 10.0


def is_log_grid(settings):
    return settings.get("GridSpacing") == "log"


def grid_edges(settings, cells=None):
    n = int(settings.get("N", 250) if cells is None else cells)
    lower, upper = float(settings.get("Lmin", 0)), float(settings.get("Lmax", 250))
    if n < 2 or not np.isfinite(upper) or not 0 <= lower < upper:
        raise ValueError("Use at least two bins and increasing nonnegative size bounds.")
    spacing = settings.get("GridSpacing", "uniform")
    if spacing == "uniform":
        return np.linspace(lower, upper, n + 1)
    if spacing != "log":
        raise ValueError(f"Unknown particle-size grid spacing {spacing!r}.")
    if settings.get("Numerics") == "matlab_pe":
        raise ValueError("Archived MATLAB PE requires a uniform size grid.")
    offset = float(settings.get("LogGridOffset", DEFAULT_LOG_OFFSET))
    if not np.isfinite(offset) or offset <= 0:
        raise ValueError("The logarithmic grid offset must be finite and positive.")
    origin = lower + offset
    span = upper - lower
    # Uniform in log(size + offset), with stable endpoints even when the offset is large.
    with np.errstate(over="ignore", invalid="ignore"):
        factor = np.logaddexp(0., np.log(span) - np.log(origin))
        edges = lower + origin * np.expm1(np.linspace(0., factor, n + 1))
    edges[0], edges[-1] = lower, upper
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Use a larger log grid offset or fewer bins for this size range.")
    return edges
