"""Inspect PBM mesh accuracy against the independent QMOM implementation."""
from time import monotonic

import numpy as np

from .distribution import cell_volumes, d_values
from .solver import model_parameters, simulate
from .validation.kinetics_sweep import pbm_values
from .validation.qmom import quadrature, solve_ccas_qmom


def _distribution_percentiles(length, density):
    if not np.isfinite(density).all() or np.maximum(density, 0).sum() <= 0:
        return np.full(3, np.nan)
    return d_values(length, np.maximum(density, 0))


def percentile_history(csd, state, process, settings, *, cancelled=None):
    cells = int(settings.get("N", 250))
    bounds = np.linspace(settings.get("Lmin", 0), settings.get("Lmax", 250), cells + 1)
    volumes = cell_volumes(bounds, process["ShapeFactor"])
    percentiles = []
    for density in state[:cells].T:
        if cancelled is not None and cancelled():
            raise InterruptedError("Size percentile calculation stopped.")
        percentiles.append(_distribution_percentiles(csd[0], density*volumes))
    return np.asarray(percentiles).T


def compare_model(model, experiment, process, solubility, settings, *, cancelled=None, progress=None):
    started = monotonic()
    parameters = model_parameters(model)
    if progress is not None:
        progress("Running the PBM on the selected mesh…")
    count, concentration, csd, state = simulate(parameters, experiment, process, solubility, settings,
                                              return_history=True, cancelled=cancelled)
    pbm_seconds = monotonic() - started
    if progress is not None:
        progress(f"PBM simulation completed in {pbm_seconds:.3f} s. Calculating comparison traces…")
    time = count[0]
    cells = int(settings.get("N", 250))
    bounds = np.linspace(settings.get("Lmin", 0), settings.get("Lmax", 250), cells + 1)
    settings = {**settings, "N": cells, "Lmin": bounds[0], "Lmax": bounds[-1]}
    values, _ = pbm_values(count, concentration, csd, state, process, settings)
    moments = np.stack([values[f"M{i}"] for i in range(6)])
    volumes = cell_volumes(bounds, process["ShapeFactor"])
    percentiles = percentile_history(csd, state, process, settings, cancelled=cancelled)
    final_volume = np.maximum(csd[1], 0) * volumes * np.diff(bounds)
    final_density = (100*final_volume/final_volume.sum()/np.diff(bounds)
                     if final_volume.sum() > 0 else np.zeros(cells))
    initial_c = (experiment["InitialConcentration"] if "InitialConcentration" in experiment
                 else np.asarray(experiment["ConcentrationData"]).reshape(2, -1)[1, 0])
    initial_api = (process["SeedMass"] + initial_c/(1-initial_c)
                   * (process["SolvMass"] + process["AntiSolvMass"]))
    temperature = np.interp(time, *np.asarray(experiment["TemperatureData"]).reshape(2, -1))
    anti = 1 - state[cells + 2]
    kelvin = temperature + 273.15
    a, b, c, d, e, f, g, h = (solubility[key] for key in "abcdefgh")
    solubility_curve = np.exp(a + b/kelvin + c*np.log(kelvin) + d*anti + e*np.exp(anti)
                              + f*anti**2 + g*anti/kelvin + h*np.exp(anti)/kelvin)
    result = {"time": time, "bounds": bounds, "length": csd[0], "settings": settings,
              "initial_api": initial_api, "temperature": temperature, "antisolvent": anti,
              "pbm_seconds": pbm_seconds,
              "solubility": solubility_curve,
              "pbm": {"moments": moments, "concentration": concentration[1],
                      "dissolved": values["dissolved_g"], "crystal": values["crystal_g"],
                      "overflow": values["overflow_g"], "percentiles": percentiles,
                      "final_density": final_density}}
    if cancelled is not None and cancelled():
        raise InterruptedError("PBM–QMOM comparison stopped.")
    if progress is not None:
        progress("Running the independent six-moment QMOM reference…")
    qmom_started = monotonic()
    try:
        reference = solve_ccas_qmom(parameters, experiment, process, solubility, settings,
                                     time=time, cancelled=cancelled)
        result["qmom_seconds"] = monotonic() - qmom_started
        qmom_percentiles = np.full((3, len(time)), np.nan)
        final_weights = np.zeros(cells)
        length_scale = max(bounds[-1]/4, 1.)
        for i in range(len(time)):
            if cancelled is not None and cancelled():
                raise InterruptedError("PBM–QMOM comparison stopped.")
            nodes, weights = quadrature(reference.moments[:, i]/length_scale**np.arange(6))
            nodes = nodes*length_scale
            volume_weights = weights*nodes**3
            total = volume_weights.sum()
            if total > 0:
                cdf = np.cumsum(volume_weights)/total
                qmom_percentiles[:, i] = nodes[np.searchsorted(cdf, [.1, .5, .9]).clip(0, len(nodes)-1)]
            if i == len(time)-1 and total > 0:
                final_weights, _ = np.histogram(nodes, bounds, weights=volume_weights/total)
        result["qmom"] = {"moments": reference.moments, "concentration": reference.concentration,
                          "dissolved": reference.dissolved_mass, "crystal": reference.crystal_mass,
                          "overflow": np.zeros_like(time), "percentiles": qmom_percentiles,
                          "final_density": 100*final_weights/np.diff(bounds)}
    except (ValueError, RuntimeError, TimeoutError, FloatingPointError) as error:
        result["qmom_error"] = str(error)
        result["qmom_seconds"] = monotonic() - qmom_started
    result["total_seconds"] = monotonic() - started
    return result
