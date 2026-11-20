"""Process and fitting defaults shared by LOOP execution and its setup preview."""
from math import isfinite
from numbers import Real


PROCESS_FIELDS = {
    "SolvMass": "Initial solvent mass (g)", "AntiSolvMass": "Initial antisolvent mass (g)",
    "SolventDensity": "Solvent density (g/cm³)", "AntiSolventDensity": "Antisolvent density (g/cm³)",
    "SolidDensity": "Crystal density (g/cm³)", "ShapeFactor": "Crystal shape factor",
    "SeedMass": "Seed mass (g)", "PowerRatio": "Power ratio",
}
PROCESS_DEFAULTS = {"SeedMass": 0., "AntiSolvMass": 0., "PowerRatio": 1.}
VALIDATION_PE_SETTINGS = {"global_iterations": 0, "population_size": 5, "local_evaluations": 40}


def require_positive_process(values, fields):
    missing = [PROCESS_FIELDS[key] for key in fields
               if not isinstance(value := values.get(key), Real) or isinstance(value, bool)
               or not isfinite(value) or value < 1e-12]
    if missing:
        raise ValueError("Enter positive values under Process inputs for " + ", ".join(missing) + ".")


def starting_process_defaults(setup, case, simulation=None):
    if simulation:
        from imed_python.demo_campaign import model_inputs

        return model_inputs(simulation["seeded"])[2]
    template = next(iter(case.get("ProcInf", [])), {})
    return {**PROCESS_DEFAULTS, **template, **setup["process"]}
