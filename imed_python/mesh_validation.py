"""Evaluate PBM solutions and solve time on successively refined size meshes."""

from time import monotonic

import numpy as np

from .comparison import initial_api_mass
from .distribution import cell_volumes, d_values, finite_volume_density
from .estimation import TRIAL_ERRORS
from .solver import model_parameters, pbm_backend_label, simulate


def mesh_counts(minimum, maximum, stages):
    if not 2 <= minimum < maximum:
        raise ValueError("Minimum bins must be at least 2 and smaller than maximum bins.")
    if not 2 <= stages <= maximum - minimum + 1:
        raise ValueError("Select at least two stages and enough distinct bin counts for the selected range.")
    return np.floor(np.linspace(minimum, maximum, stages) + 0.5).astype(int)


def validate_meshes(case, model_index, experiment_index, counts, *, cancelled=None, progress=None):
    settings = case["pbmsetting"]
    backend = pbm_backend_label(settings)
    uses_jax = settings.get("Numerics") == "jax"
    lower, upper = float(settings["Lmin"]), float(settings["Lmax"])
    parameters = model_parameters(case["ModelParameter"][model_index])
    experiment, process = case["ExpData"][experiment_index], case["ProcInf"][experiment_index]
    shape = process["ShapeFactor"]
    initial_mass = initial_api_mass(experiment, process)
    runs = []

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError("Mesh validation stopped.")

    for index, bins in enumerate(counts, 1):
        check_cancelled()
        bins = int(bins)
        mesh = {**settings, "N": bins}
        inputs = parameters, experiment, process, case["Solubility"], mesh
        row = {"bins": bins, "pbm_seconds": np.nan, "density": np.empty(0),
               "percentiles": np.full(3, np.nan), "mass_error_ppm": np.nan, "status": "Complete"}
        step = f"Mesh {index}/{len(counts)} · {bins:,} bins"
        try:
            if uses_jax:
                if progress is not None:
                    progress(f"{step} · JAX warm-up")
                simulate(*inputs, return_history=True, cancelled=cancelled)
                check_cancelled()
            if progress is not None:
                progress(f"{step} · Timing the PBM solve")
            started = monotonic()
            _, _, csd, state = simulate(*inputs, return_history=True, cancelled=cancelled)
            row["pbm_seconds"] = monotonic() - started
            bounds = np.linspace(lower, upper, bins + 1)
            volumes = cell_volumes(bounds, shape)
            density = finite_volume_density(csd[1], bounds, shape)
            row["density"] = density
            if density.any():
                row["percentiles"] = d_values(csd[0], density)
            else:
                row["status"] = "No predicted particle volume"
            crystals = (volumes * np.diff(bounds)) @ state[:bins] * process["SolidDensity"] / 1e12
            inventory = state[bins] * state[bins + 1] + crystals + state[bins + 5]
            if initial_mass > 0:
                row["mass_error_ppm"] = float(1e6 * np.max(np.abs(inventory - initial_mass)) / initial_mass)
        except TRIAL_ERRORS as error:
            row["status"] = str(error)
        runs.append(row)
    check_cancelled()
    return {"experiment_index": experiment_index, "backend": backend, "uses_jax": uses_jax,
            "bounds": [lower, upper], "settings": dict(settings), "runs": runs}
