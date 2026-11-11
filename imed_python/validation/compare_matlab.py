"""Compare the IMED PBM against predictions encoded in an archived MATLAB PE fit."""
import argparse
import csv
from pathlib import Path

import numpy as np

from ..distribution import volume_distribution
from ..solver import model_parameters, read_model, simulate


def recover_predictions(case, model):
    """Invert weighted residuals only for the fitted prefix of the experiment list."""
    weights = case["objweightfactor"]
    if weights["Terms"].casefold() == "csd" or min(weights[f"W{i}"] for i in range(1, 4)) <= 0:
        raise ValueError("Reconstruction requires positive count, concentration and D-value weights")
    experiments = case["ExpData"]
    residuals = np.asarray(model["resid"]).reshape(-1)
    lengths = [exp["CountsData"].shape[1] + exp["ConcentrationData"].shape[1] + 3
               for exp in experiments]
    prefix = np.flatnonzero(np.cumsum(lengths) == residuals.size)
    if len(prefix) != 1 or not np.isclose(residuals @ residuals, model["resnorm"]):
        raise ValueError("The saved residuals do not match a complete fitted experiment prefix")
    fitted_count = int(prefix[0]) + 1
    offset = 0
    references = []
    scalar_score = 0.
    for exp in experiments[:fitted_count]:
        observed = (exp["CountsData"][1], exp["ConcentrationData"][1], exp["FinalD10D50D90"])
        data = []
        for i, values in enumerate(observed, 1):
            if i == 1:
                values = values / values.max()
            size = len(values)
            scale = np.sqrt(weights[f"W{i}"] / (fitted_count * size))
            data.append(values + residuals[offset:offset + size] / scale)
            offset += size
        if (not all(np.isfinite(values).all() and (values >= -1e-8).all() for values in data)
                or not np.isclose(data[0].max(), 1, atol=1e-8, rtol=0)
                or np.any(np.diff(data[2]) < 0)):
            raise ValueError("Reconstructed traces failed normalization or physical checks")
        differences = (data[0] / data[0][-1] - observed[0] / observed[0][-1],
                       data[1] - observed[1], data[2] - observed[2])
        scalar_score += sum(weights[f"W{i}"] * np.mean(delta**2)
                            for i, delta in enumerate(differences, 1)) / fitted_count
        references.append(data)
    if "SSR" in model and not np.isclose(scalar_score, model["SSR"], atol=1e-10, rtol=1e-8):
        raise ValueError("Reconstructed traces do not reproduce the independently saved MATLAB SSR")
    return references


def compare(case, modes):
    rows, trajectories = [], []
    for model_number, model in enumerate(case["ModelParameter"], 1):
        references = recover_predictions(case, model)
        pm = model_parameters(model)
        for i, (experiment, process) in enumerate(zip(case["ExpData"], case["ProcInf"])):
            reference = references[i] if i < len(references) else None
            measured_count = experiment["CountsData"]
            measured_conc = experiment["ConcentrationData"]
            for mode in modes:
                settings = {**case["pbmsetting"], "Numerics": mode}
                row = {"model": model_number, "experiment": i + 1, "numerics": mode,
                       "matlab_reference": reference is not None, "status": "completed"}
                try:
                    count, concentration, csd = simulate(pm, experiment, process, case["Solubility"], settings)
                    _, percentiles = volume_distribution(csd, process, settings)
                    observed_times = (measured_count[0], measured_conc[0])
                    count_interpolated = np.interp(observed_times[0], *count, left=np.nan, right=np.nan)
                    c = np.interp(observed_times[1], *concentration, left=np.nan, right=np.nan)
                    count_interpolated /= count_interpolated.max()
                    row.update(python_final_concentration=c[-1], measured_final_concentration=measured_conc[1, -1])
                    if percentiles is not None:
                        row.update({f"python_D{p}_um": value for p, value in zip((10, 50, 90), percentiles)})
                    if reference is not None:
                        ref_count, ref_c, ref_d = reference
                        row.update(normalized_count_rmse=np.sqrt(np.mean((count_interpolated - ref_count)**2)),
                                   concentration_rmse=np.sqrt(np.mean((c - ref_c)**2)),
                                   matlab_final_concentration=ref_c[-1])
                        row.update({f"matlab_D{p}_um": value for p, value in zip((10, 50, 90), ref_d)})
                        if percentiles is not None:
                            row["max_d_error_um"] = np.max(abs(percentiles - ref_d))
                    else:
                        ref_count = np.full(len(count_interpolated), np.nan)
                        ref_c = np.full(len(c), np.nan)
                    for channel, t, simulated, matlab, measured in (
                        ("count_normalized", observed_times[0], count_interpolated, ref_count,
                         measured_count[1] / measured_count[1].max()),
                        ("concentration", observed_times[1], c, ref_c, measured_conc[1]),
                    ):
                        trajectories.extend(dict(model=model_number, experiment=i + 1, numerics=mode,
                                                 channel=channel, time_s=x, python=y, matlab=z, measured=w)
                                            for x, y, z, w in zip(t, simulated, matlab, measured))
                except (ValueError, RuntimeError, TimeoutError, FloatingPointError) as error:
                    row["status"] = str(error)
                rows.append(row)
                print(model_number, i + 1, mode, row["status"],
                      {key: round(row[key], 6) for key in ("normalized_count_rmse", "concentration_rmse",
                                                           "max_d_error_um") if key in row}, flush=True)
    return rows, trajectories


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["conservative", "matlab_pe"],
                        choices=["conservative", "matlab_pe"])
    args = parser.parse_args()
    rows, trajectories = compare(read_model(args.case), args.modes)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", rows)
    write_csv(args.output / "trajectories.csv", trajectories)
    return int(any(row["status"] != "completed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
