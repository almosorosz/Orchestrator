"""Settings shared between a PE dialog and an IMED case."""
import math

from .size_data import FINAL_SIZE_TERM, TREND_SIZE_TERM

# MATLAB ControlPanel.m and ParameterEst/+utils/pe_build_default_settings.m.
PBM_DEFAULTS = {
    "Lmin": 0., "Lmax": 250., "N": 250, "RelTol": 1e-5, "AbsTol": 1e-5,
    "MaxIterationTime": 30.,
}
WEIGHT_DEFAULTS = {"W1": 0., "W2": 0., "W3": 0., "Terms": "CSD"}
PYFIT_SETTINGS_KEY = "pyfitsettings"
PE_PBM_KEY = "pepbmsettings"
PE_WEIGHTS_KEY = "peweights"
PYFIT_DEFAULTS = {
    "global_iterations": 20,
    "population_size": 200,
    "global_tolerance": 0.01,
    "mutation_min": 0.5,
    "mutation_max": 1.0,
    "recombination": 0.7,
    "local_evaluations": 1500,
}
PE_PBM_FIELDS = (*PBM_DEFAULTS, "MaxSteps", "UseArchivedTolerances")


def pbm_for_pe(saved):
    """Fill missing PE mesh values using the selected backend's defaults."""
    saved = saved if isinstance(saved, dict) else {}
    defaults = dict(PBM_DEFAULTS)
    if saved.get("Numerics") == "jax":
        defaults.update(RelTol=1e-7, AbsTol=1e-9, MaxSteps=100_000)
    return {**defaults, **saved}


def effective_pe_pbm(case):
    pbm = pbm_for_pe(case.get("pbmsetting"))
    saved = case.get(PE_PBM_KEY)
    if isinstance(saved, dict):
        pbm.update({key: saved[key] for key in PE_PBM_FIELDS if key in saved})
    return pbm


def pe_mesh(pbm):
    return {key: pbm[key] for key in PE_PBM_FIELDS if key in pbm}


def pe_file(pbm, optimizer, weights):
    return {
        "format": "crystalloop-pe", "version": 1,
        "pbm": pe_mesh(pbm),
        "optimizer": {key: optimizer[key] for key in PYFIT_DEFAULTS},
        "weights": dict(weights),
    }


def read_pe_file(data):
    if not isinstance(data, dict) or data.get("format") != "crystalloop-pe" or data.get("version") != 1:
        raise ValueError("Choose a CrystalLoop PE settings file.")

    def numbers(group, allowed):
        values = data.get(group)
        if not isinstance(values, dict) or set(values) - set(allowed):
            raise ValueError(f"Invalid PE {group}.")
        return values

    def valid(value, minimum, *, strict=False):
        return (type(value) in (int, float) and math.isfinite(value)
                and (value > minimum if strict else value >= minimum))

    pbm = numbers("pbm", PE_PBM_FIELDS)
    if any(not valid(pbm.get(key, default), 0, strict=key != "Lmin")
           for key, default in PBM_DEFAULTS.items()):
        raise ValueError("Invalid PE mesh, tolerances or simulation timeout.")
    if (pbm.get("Lmax", 250) <= pbm.get("Lmin", 0)
            or type(pbm.get("N", 250)) is not int or pbm.get("N", 250) < 2
            or type(pbm.get("MaxSteps", 1)) is not int or pbm.get("MaxSteps", 1) < 1
            or type(pbm.get("UseArchivedTolerances", False)) is not bool):
        raise ValueError("Invalid PE mesh, tolerances or step limit.")
    optimizer = {**PYFIT_DEFAULTS, **numbers("optimizer", PYFIT_DEFAULTS)}
    minima = {"global_iterations": 0, "population_size": 5, "local_evaluations": 1}
    if (any(not valid(value, minima.get(key, 0)) for key, value in optimizer.items())
            or any(type(optimizer[key]) is not int for key in minima)
            or not optimizer["mutation_min"] <= optimizer["mutation_max"] < 2
            or optimizer["recombination"] > 1):
        raise ValueError("Invalid PE optimizer settings.")
    weights = {**WEIGHT_DEFAULTS, **numbers("weights", WEIGHT_DEFAULTS)}
    if (any(not valid(weights[key], 0) or weights[key] > 10000 for key in ("W1", "W2", "W3"))
            or weights["Terms"] not in ("CSD", FINAL_SIZE_TERM, TREND_SIZE_TERM)):
        raise ValueError("Invalid PE response weights or size term.")
    return pbm, optimizer, weights
