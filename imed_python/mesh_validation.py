"""Evaluate PBM solutions and solve time on successively refined size meshes."""

import numpy as np

from .comparison import initial_api_mass, timed_pbm
from .distribution import cell_volumes, d_values, finite_volume_density
from .estimation import TRIAL_ERRORS
from .mesh_grid import grid_edges, is_log_grid
from .solver import model_parameters, pbm_backend_label


def mesh_counts(minimum, maximum, stages):
    if not 2 <= minimum < maximum:
        raise ValueError("Minimum bins must be at least 2 and smaller than maximum bins.")
    if not 2 <= stages <= maximum - minimum + 1:
        raise ValueError("Select at least two stages and enough distinct bin counts for the selected range.")
    return np.floor(np.linspace(minimum, maximum, stages) + 0.5).astype(int)


def validate_meshes(case, model_index, experiment_index, counts, *, cancelled=None, progress=None,
                    budget_expired=None):
    settings = case["pbmsetting"]
    backend = pbm_backend_label(settings)
    uses_jax = settings.get("Numerics") == "jax"
    lower, upper = float(settings["Lmin"]), float(settings["Lmax"])
    parameters = model_parameters(case["ModelParameter"][model_index])
    experiment, process = case["ExpData"][experiment_index], case["ProcInf"][experiment_index]
    shape = process["ShapeFactor"]
    initial_mass = initial_api_mass(experiment, process)
    runs = []

    def interrupted():
        if cancelled is None or not cancelled():
            return False
        if budget_expired is not None and budget_expired():
            return True
        raise InterruptedError("Mesh validation stopped.")

    for index, bins in enumerate(counts, 1):
        if interrupted():
            break
        bins = int(bins)
        mesh = {**settings, "N": bins}
        inputs = parameters, experiment, process, case["Solubility"], mesh
        row = {"bins": bins, "pbm_seconds": np.nan, "density": np.empty(0),
               "percentiles": np.full(3, np.nan), "mass_error_ppm": np.nan, "status": "Complete"}
        step = f"Mesh {index}/{len(counts)} · {bins:,} bins"
        try:
            (_, _, csd, state), row["pbm_seconds"] = timed_pbm(
                *inputs, cancelled=cancelled, progress=progress, label=step)
            bounds = grid_edges(mesh)
            volumes = cell_volumes(bounds, shape)
            density = finite_volume_density(csd[1], bounds, shape)
            row["density"] = density
            if density.any():
                row["percentiles"] = d_values(csd[0], density, edges=bounds if is_log_grid(mesh) else None)
            else:
                row["status"] = "No predicted particle volume"
            crystals = (volumes * np.diff(bounds)) @ state[:bins] * process["SolidDensity"] / 1e12
            inventory = state[bins] * state[bins + 1] + crystals + state[bins + 5]
            if initial_mass > 0:
                row["mass_error_ppm"] = float(1e6 * np.max(np.abs(inventory - initial_mass)) / initial_mass)
        except InterruptedError:
            if interrupted():
                break
            raise
        except TRIAL_ERRORS as error:
            row["status"] = str(error)
        if interrupted():
            break
        runs.append(row)
    expired = interrupted()
    if expired and not runs:
        raise TimeoutError("Time budget reached before the first mesh finished.")
    return {"experiment_index": experiment_index, "backend": backend, "uses_jax": uses_jax,
            "bounds": [lower, upper], "settings": dict(settings), "runs": runs,
            "planned_bins": np.asarray(counts, dtype=int), "budget_expired": expired}
