"""Process and fitting defaults shared by LOOP execution and its setup preview."""
from imed_python.demo_campaign import model_inputs


PROCESS_DEFAULTS = {"SeedMass": 0., "AntiSolvMass": 0., "PowerRatio": 1.}
LOOP_WEIGHT_DEFAULTS = (0, 1, 1)
VALIDATION_PE_SETTINGS = {"global_iterations": 0, "population_size": 5, "local_evaluations": 40}


def starting_process_defaults(setup, case, simulation=None):
    if simulation:
        return model_inputs(simulation["seeded"])[2]
    template = next(iter(case.get("ProcInf", [])), {})
    return {**PROCESS_DEFAULTS, **template, **setup["process"]}
