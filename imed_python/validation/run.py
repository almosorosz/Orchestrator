"""Run the offline FVM/QMOM validation matrix and write reproducible CSVs."""
import argparse
import csv
from pathlib import Path

import numpy as np

from ..distribution import cell_volumes
from ..solver import simulate
from .cases import Case
from .qmom import solve_reference


LIMITS = {"M0": 0.025, "M1": 0.025, "M2": 0.025, "M3": 0.025, "M4": 0.04,
          "M5": 0.05, "count": 0.025, "concentration": 0.001,
          "mass_balance": 0.01}


def relative_error(actual, expected):
    return float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))


def compare(case, cells, reference):
    params, experiment, process, solubility, settings = case.inputs(cells)
    count, concentration, csd, state = simulate(
        params, experiment, process, solubility, settings, return_history=True)
    edges = np.linspace(0, case.length_max, cells + 1)
    number = state[:cells] * np.diff(edges)[:, None]
    basis = (csd[0] / case.length_scale)**np.arange(6)[:, None]
    basis[3] = cell_volumes(edges, case.shape) / (case.shape * case.length_scale**3)
    moments = basis @ number / case.number_scale
    solid_mass = cell_volumes(edges, case.shape) @ number * case.density / 1e12
    inventory = solid_mass + state[cells] * state[cells + 1]
    errors = {f"M{k}": relative_error(moments[k], reference.moments[k]) for k in range(6)}
    errors["count"] = relative_error(count[1], reference.count)
    errors["concentration"] = relative_error(concentration[1], reference.concentration)
    errors["mass_balance"] = float(np.max(np.abs(inventory - inventory[0])) / solid_mass[0])
    breakup = state[cells + 4, -1] / case.number_scale
    aggregation = (state[cells + 3, -1] + state[cells + 4, -1]) / case.number_scale
    exercised = not case.breakage or min(aggregation, breakup,
                                         reference.aggregation_events[-1],
                                         reference.breakup_events[-1]) > 1e-4
    row = {"profile": case.profile, "mechanisms": case.mechanisms, "cells": cells,
           **errors, "FVM_aggregation": aggregation, "QMOM_aggregation": reference.aggregation_events[-1],
           "FVM_breakage": breakup, "QMOM_breakage": reference.breakup_events[-1],
           "mechanisms_exercised": exercised,
           "passed": exercised and all(errors[key] <= limit for key, limit in LIMITS.items())}
    values = {"count": (count[1], reference.count),
              "concentration": (concentration[1], reference.concentration),
              **{f"M{k}": (moments[k], reference.moments[k]) for k in range(6)}}
    trace = [
        {"profile": case.profile, "mechanisms": case.mechanisms, "cells": cells, "time_s": t,
         **{f"{name}_{method}": data[point]
            for name, pair in values.items()
            for method, data in zip(("FVM", "QMOM"), pair)}}
        for point, t in enumerate(count[0])
    ]
    return row, trace


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=int, default=[80, 160, 320])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, trace, reference_checks = [], [], []
    for profile in ("cooling", "antisolvent", "combined"):
        for mechanisms in ("growth", "nucleation", "full"):
            case = Case(profile, mechanisms)
            time = np.linspace(0, case.duration, 100)
            reference = solve_reference(case, time)
            refined = solve_reference(case, time, rtol=1e-11, atol=1e-13)
            reference_checks.append({
                "profile": profile, "mechanisms": mechanisms,
                **{f"M{k}": relative_error(reference.moments[k], refined.moments[k])
                   for k in range(6)},
                "count": relative_error(reference.count, refined.count),
                "concentration": relative_error(reference.concentration, refined.concentration),
            })
            for cells in args.cells:
                row, history = compare(case, cells, reference)
                rows.append(row)
                trace.extend(history)
                print(profile, mechanisms, cells, "PASS" if row["passed"] else "FAIL",
                      " ".join(f"{key}={row[key]:.4g}" for key in LIMITS),
                      f"breakage={row['FVM_breakage']:.4g}/{row['QMOM_breakage']:.4g}", flush=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / "summary.csv", rows)
        write_csv(args.output / "trajectories.csv", trace)
        write_csv(args.output / "reference_refinement.csv", reference_checks)
    finest = max(args.cells)
    reference_failed = any(max(row[key] for key in (*[f"M{k}" for k in range(6)],
                                                    "count", "concentration")) > 1e-5
                           for row in reference_checks)
    return int(reference_failed or any(not row["passed"] for row in rows if row["cells"] == finest))


if __name__ == "__main__":
    raise SystemExit(main())
