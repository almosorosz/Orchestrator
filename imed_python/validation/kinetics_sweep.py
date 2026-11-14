"""Run an offline kinetic sweep against the production IMED PBM and QMOM."""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..distribution import cell_volumes
from ..solver import simulate
from .cases import Case
from .qmom import solve_ccas_qmom
from .run import relative_error, write_csv


@dataclass(frozen=True)
class Kinetics:
    profile: str = "combined"
    kg: float = 0.65
    kp: float = 1.3e4
    ks: float = 5e5
    ka: float = 5e-8
    kbr: float = 0.12
    kernel: int = 1
    growth_size_factor: float = 0.
    aggregation_growth_power: float = 0.
    breakage_volume_power: float = 0.
    kd: float = 0.
    daughter_center: float = 1400.
    daughter_width: float = 360.
    secondary_moment: int = 0  # 0 volume, 1 surface area
    length_max: float = 65.

    def inputs(self, cells):
        case = Case(self.profile, "full", growth=self.kg, length_max=self.length_max,
                    daughter_center=self.daughter_center, daughter_width=self.daughter_width)
        parameters, experiment, process, solubility, settings = case.inputs(cells)
        for index, coefficient in ((0, self.kp), (5, self.ks), (18, self.kd),
                                   (24, self.ka), (28, self.kbr)):
            parameters[index] = np.log10(coefficient) if coefficient else -np.inf
        parameters[6:9] = 1, 1, self.secondary_moment
        parameters[14:16] = self.growth_size_factor, 1
        parameters[19] = 1
        parameters[25:27] = self.aggregation_growth_power, self.kernel
        parameters[29] = self.breakage_volume_power
        return parameters, experiment, process, solubility, settings


BASE = Kinetics()
CASES = (
    ("K01", replace(BASE, kp=2e3)),
    ("K02", replace(BASE, kp=6e4)),
    ("K03", replace(BASE, ks=5e4)),
    ("K04", replace(BASE, ks=3e6)),
    ("K05", replace(BASE, ka=8e-9)),
    ("K06", replace(BASE, ka=4e-7)),
    ("K07", replace(BASE, kbr=0.015)),
    ("K08", replace(BASE, kbr=0.6)),
    ("K09", Kinetics("cooling", kg=0.45, kp=0, ks=0, ka=0, kbr=0)),
    ("K10", Kinetics("cooling", kg=0.9, kp=3e4, ks=0, ka=0, kbr=0, kd=0.02)),
    ("K11", Kinetics("cooling", kg=0.75, kp=0, ks=2e6, ka=2e-8, kbr=0)),
    ("K12", Kinetics("cooling", kp=0, ks=0, ka=1e-8, kbr=0.18, kernel=2, kd=0.06)),
    ("K13", Kinetics("antisolvent", kg=0.8, kp=6e3, ks=0, ka=2e-7, kbr=0.2)),
    ("K14", Kinetics("antisolvent", kg=0.45, kp=0, ks=4e4, ka=9e-8, kbr=0.3,
                      secondary_moment=1)),
    ("K15", Kinetics("antisolvent", kg=1.0, kp=4e4, ks=1e6, ka=3e-12, kbr=0.1,
                      kernel=3, growth_size_factor=0.02, aggregation_growth_power=0.5)),
    ("K16", Kinetics("antisolvent", kg=0.55, kp=9e3, ks=8e5, ka=4e-12, kbr=0.15,
                      kernel=4, daughter_center=900., daughter_width=170.)),
    ("K17", Kinetics("combined", kg=0.5, kp=6e3, ks=2e5, ka=4e-12, kbr=0.1,
                    kernel=5)),
    ("K18", Kinetics("combined", kg=0.85, kp=8e3, ks=1e6, ka=1.2e-7, kbr=0.025,
                    breakage_volume_power=0.2, daughter_center=2500., daughter_width=500.)),
    ("K19", Kinetics("combined", kg=0.7, kp=0, ks=0, ka=6e-8, kbr=0.08,
                    growth_size_factor=0.03, aggregation_growth_power=0.5, kd=0.1)),
    ("K20", Kinetics("combined", kg=1.1, kp=4e4, ks=2e6, ka=1.5e-7, kbr=0.35,
                    growth_size_factor=0.015, aggregation_growth_power=1.,
                    daughter_center=950., daughter_width=210.)),
)

MOMENTS = tuple(f"M{order}" for order in range(6))
LIMITS = dict.fromkeys(MOMENTS[:4], 0.025) | {
    "M4": 0.04, "M5": 0.05, "count": 0.025, "mean_size": 0.025,
    "concentration": 0.001, "dissolved_g": 0.001,
}
MASS_LIMIT = 1e-6  # relative to initial dissolved + seed API
QMOM_REFINEMENT_LIMIT = 1e-5
EVENT_FRACTION = 1e-5
OVERFLOW_LIMIT = 1e-4  # relative to peak resolved crystal mass


def reference_values(reference, process):
    liquid_volume = (process["SolvMass"] / process["SolventDensity"]
                     + (reference.liquid_mass - process["SolvMass"]) / process["AntiSolventDensity"])
    return dict(zip(MOMENTS, reference.moments)) | {
        "count": reference.number / liquid_volume, "mean_size": reference.mean_size,
        "concentration": reference.concentration, "dissolved_g": reference.dissolved_mass,
        "crystal_g": reference.crystal_mass, "overflow_g": np.zeros_like(reference.time),
    }


def pbm_values(count, concentration, csd, state, process, settings):
    cells = settings["N"]
    edges = np.linspace(settings["Lmin"], settings["Lmax"], cells + 1)
    numbers = state[:cells] * np.diff(edges)[:, None]
    volume = cell_volumes(edges, process["ShapeFactor"])
    moments = csd[0]**np.arange(6)[:, None] @ numbers
    moments[3] = volume @ numbers / process["ShapeFactor"]
    values = dict(zip(MOMENTS, moments)) | {
        "count": count[1], "mean_size": moments[1] / moments[0],
        "concentration": concentration[1], "dissolved_g": state[cells] * state[cells + 1],
        "crystal_g": moments[3] * process["ShapeFactor"] * process["SolidDensity"] / 1e12,
        "overflow_g": state[cells + 5],
    }
    return values, numbers


def inventory(values):
    return values["dissolved_g"] + values["crystal_g"] + values["overflow_g"]


def run_case(case_id, kinetics, meshes, *, app_tolerances=False):
    reference_inputs = kinetics.inputs(max(meshes))
    _, experiment, process, _, _ = reference_inputs
    concentration = experiment["ConcentrationData"][1][0]
    initial_api = (process["SeedMass"] + concentration / (1 - concentration)
                   * (process["SolvMass"] + process["AntiSolvMass"]))
    reference = solve_ccas_qmom(*reference_inputs, rtol=1e-10, atol=1e-12)
    standard = solve_ccas_qmom(*reference_inputs)
    expected = reference_values(reference, process)
    standard_values = reference_values(standard, process)
    refinement = max(relative_error(standard_values[key], expected[key]) for key in LIMITS)
    qmom_balance = inventory(expected) - initial_api
    traces, rows = [], []
    for cells in meshes:
        inputs = kinetics.inputs(cells)
        if app_tolerances:
            for key in ("RelTol", "AbsTol", "MaxStep"):
                del inputs[-1][key]
        count, conc, csd, state = simulate(*inputs, return_history=True)
        actual, numbers = pbm_values(count, conc, csd, state, process, inputs[-1])
        errors = {key: relative_error(actual[key], expected[key]) for key in LIMITS}
        balance = inventory(actual) - initial_api
        agg = state[cells + 3, -1] + state[cells + 4, -1]
        br = state[cells + 4, -1]
        event_limit = EVENT_FRACTION * reference.number[0]
        agg_exercised = not kinetics.ka or min(agg, reference.aggregation_events[-1]) > event_limit
        br_exercised = not kinetics.kbr or min(br, reference.breakup_events[-1]) > event_limit
        negative_number_fraction = float(np.max(np.maximum(-numbers, 0).sum(axis=0))
                                         / reference.number[0])
        finite = all(np.isfinite(values).all() for values in (*actual.values(), *expected.values()))
        physical = (finite and np.all((0 <= actual["concentration"])
                                      & (actual["concentration"] < 1))
                    and np.min(actual["overflow_g"]) >= -64 * np.finfo(float).eps * initial_api
                    and negative_number_fraction <= MASS_LIMIT)
        boundary_comparable = (np.max(actual["overflow_g"])
                               <= OVERFLOW_LIMIT * np.max(actual["crystal_g"]))
        mass_passed = (max(np.max(np.abs(balance)), np.max(np.abs(qmom_balance)))
                       <= MASS_LIMIT * initial_api)
        passed = (physical and mass_passed and boundary_comparable and agg_exercised and br_exercised
                  and refinement <= QMOM_REFINEMENT_LIMIT
                  and all(errors[key] <= limit for key, limit in LIMITS.items()))
        rows.append({"case": case_id, **vars(kinetics), "cells": cells,
                     "tolerances": "app defaults" if app_tolerances else "benchmark",
                     "status": "PASS" if passed else "FAIL",
                     **{f"{key}_error_pct": 100 * error for key, error in errors.items()},
                     "initial_api_g": initial_api, "pbm_mass_error_g": np.max(np.abs(balance)),
                     "qmom_mass_error_g": np.max(np.abs(qmom_balance)),
                     "pbm_mass_error_ppm": np.max(np.abs(balance)) / initial_api * 1e6,
                     "qmom_mass_error_ppm": np.max(np.abs(qmom_balance)) / initial_api * 1e6,
                     "pbm_overflow_g": actual["overflow_g"][-1],
                     "negative_number_ppm": negative_number_fraction * 1e6,
                     "qmom_refinement_pct": 100 * refinement,
                     "pbm_agglomeration_events": agg, "qmom_agglomeration_events": reference.aggregation_events[-1],
                     "pbm_breakage_count_increase": br, "qmom_breakage_count_increase": reference.breakup_events[-1],
                     "agg_exercised": agg_exercised, "breakage_exercised": br_exercised,
                     "physical": physical, "mass_passed": mass_passed,
                     "boundary_comparable": boundary_comparable})
        traces.extend({"case": case_id, "cells": cells, "time_s": time,
                       **{f"{key}_{method}": values[key][i]
                          for method, values in (("pbm", actual), ("qmom", expected))
                          for key in actual},
                       "pbm_inventory_error_g": balance[i], "qmom_inventory_error_g": qmom_balance[i]}
                      for i, time in enumerate(reference.time))
        print(f"{case_id} {cells} {rows[-1]['status']} "
              f"moment={max(errors[key] for key in MOMENTS):.3%} "
              f"mass={rows[-1]['pbm_mass_error_ppm']:.3g} ppm "
              f"agg={agg:.3g} breakage={br:.3g}", flush=True)
        if not passed:
            print(f"  physical={physical} comparable={boundary_comparable} "
                  f"QMOM refinement={refinement:.3g} "
                  + " ".join(f"{key}={value:.3%}" for key, value in errors.items()), flush=True)
    return rows, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[250])
    parser.add_argument("--length-max", type=float, help="Override the upper size bound in µm")
    parser.add_argument("--aggregation-scale", type=float, default=1.,
                        help="Multiply the selected agglomeration coefficients by this factor")
    parser.add_argument("--app-tolerances", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=[case_id for case_id, _ in CASES])
    args = parser.parse_args()
    if not np.isfinite(args.aggregation_scale) or args.aggregation_scale < 0:
        parser.error("--aggregation-scale must be finite and nonnegative")
    rows, traces, failures = [], [], []
    for case_id, kinetics in CASES:
        if args.cases and case_id not in args.cases:
            continue
        if args.length_max is not None:
            kinetics = replace(kinetics, length_max=args.length_max)
        if args.aggregation_scale != 1.:
            kinetics = replace(kinetics, ka=kinetics.ka * args.aggregation_scale)
        try:
            results, trajectories = run_case(case_id, kinetics, args.cells,
                                             app_tolerances=args.app_tolerances)
        except (ValueError, RuntimeError, TimeoutError, FloatingPointError) as error:
            failures.append({"case": case_id, "error": str(error)})
            print(f"{case_id} ERROR {error}", flush=True)
        else:
            rows.extend(results)
            traces.extend(trajectories)
    args.output.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(args.output / "kinetics_summary.csv", rows)
        write_csv(args.output / "kinetics_trajectories.csv", traces)
    if failures:
        write_csv(args.output / "kinetics_failures.csv", failures)
    return int(bool(failures) or any(row["status"] != "PASS" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
