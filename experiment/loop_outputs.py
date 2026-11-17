"""Write inspectable model inputs and results alongside each campaign manifest."""
from csv import writer
from shutil import copyfile

import numpy as np
import pandas as pd

from bo.profiles import CHANNELS
from imed_python.parameter_labels import PARAMETERS


MEASUREMENTS = {
    "TemperatureData": "°C", "ConcentrationData": "g/g solution", "CountsData": "count",
    "D10Data": "µm", "D50Data": "µm", "D90Data": "µm", "FlowRateData": "g/s",
    "AntisolventScaleData": "g", "AddedAntisolventData": "g",
}


def saved_recordings(target, runs, report):
    for _, _, file in runs:
        report(f"Copying completed recording {file.parent.name} into MODEL inputs")
        destination = target / "inputs" / file.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("run.json", "planned.csv", "execution.csv"):
            copyfile(file.with_name(filename), destination / filename)


def imed_inputs(target, case):
    directory = target / "imed"
    directory.mkdir(exist_ok=True)
    process_rows = []
    with (directory / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        csv = writer(stream)
        csv.writerow(("experiment", "signal", "unit", "time_s", "value"))
        for index, (experiment, process) in enumerate(zip(case["ExpData"], case["ProcInf"]), 1):
            name = str(experiment.get("Name", f"Experiment {index}"))
            process_rows.append({"experiment": index, "name": name, "recording": experiment.get("RecordingFile", ""),
                                 "recipe_source": experiment.get("RecipeSource", ""),
                                 "recipe_stage": experiment.get("RecipeStage", ""),
                                 "recipe_reason": experiment.get("RecipeReason", ""),
                                 "recipe_model_folder": experiment.get("RecipeModelFolder", ""),
                                 **{key: value for key, value in process.items() if np.isscalar(value)}})
            for signal, unit in MEASUREMENTS.items():
                trace = np.asarray(experiment.get(signal, []))
                if trace.ndim == 2 and trace.shape[0] == 2:
                    csv.writerows((index, signal, unit, time, value) for time, value in zip(*trace))
    pd.DataFrame(process_rows).to_csv(directory / "process_inputs.csv", index=False)


def imed_fits(target, models, fits, assessment):
    directory = target / "imed"
    summary, parameters = [], []
    for index, fit in fits.items():
        model = models[index]
        name = model.get("Name", f"Model {index + 1}")
        result = assessment[index]
        summary.append({"model": index + 1, "name": name, "converged": fit.converged, "ssr": fit.ssr,
                        "aic": result["total_aic"], "aic_rank": result["rank"], "delta_aic": result["delta_aic"],
                        "f_value": result["f_value"], "f_limit": result["f_limit"], "f_pass": result["f_pass"],
                        "t_pass": result["t_pass"], "survives": not result["reasons"],
                        "reason": "; ".join(result["reasons"])})
        active = np.flatnonzero(np.asarray(model["paraset"])[:, 0] == 1)
        for parameter, value, statistic in zip(active, fit.parameters, result["t_statistics"]):
            symbol, description = PARAMETERS[parameter]
            parameters.append({"model": index + 1, "name": name, "parameter_index": parameter + 1,
                               "parameter": symbol, "description": description, "estimate": value,
                               "t_statistic": statistic, "t_limit": result["t_limit"],
                               "t_pass": bool(statistic >= result["t_limit"])})
        pd.DataFrame(fit.covariance).to_csv(directory / f"model_{index + 1}_covariance.csv", index=False, header=False)
    pd.DataFrame(summary).to_csv(directory / "pe_models.csv", index=False)
    pd.DataFrame(parameters).to_csv(directory / "pe_parameters.csv", index=False)


def bo_inputs(target, state):
    directory = target / "bo"
    directory.mkdir(exist_ok=True)
    selected = set(state["selected"])
    rows = [{"experiment": i + 1, "name": run["name"], "selected": i in selected,
             "recipe_stage": run.get("recipe_stage", ""), **run["metrics"]}
            for i, run in enumerate(state["runs"])]
    pd.DataFrame(rows).to_csv(directory / "observations.csv", index=False)
    with (directory / "training_profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        csv = writer(stream)
        csv.writerow(("experiment", "channel", "unit", "time_s", "value", "step"))
        for index, run in enumerate(state["runs"], 1):
            for channel, trace in run["profiles"].items():
                unit = CHANNELS[channel][1]
                csv.writerows((index, channel, unit, time, value, trace["step"])
                              for time, value in zip(*trace["points"]))


def modeled_recipe(target, profiles, campaign, predictions=None):
    rows = [{"channel": channel, "unit": CHANNELS[channel][1], "time_s": time, "value": value, "step": profile.step}
            for channel, profile in profiles.items() for time, value in zip(profile.time, profile.values)]
    pd.DataFrame(rows).to_csv(target / "recommended_profile.csv", index=False)
    campaign.experiments[0].to_frame().to_csv(target / "device_targets.csv", index=False)
    if predictions is not None:
        rows = [{"output": key, "mean": mean, "standard_deviation": sd}
                for key, (mean, sd) in predictions.items()]
        pd.DataFrame(rows).to_csv(target / "bo" / "predictions.csv", index=False)
