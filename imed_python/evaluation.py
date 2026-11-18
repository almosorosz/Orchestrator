"""Measured-versus-predicted summaries across the loaded experiments."""
from time import monotonic

import numpy as np

from .distribution import volume_distribution
from .estimation import TRIAL_ERRORS, series
from .probabilistic import observed_percentiles
from .solver import model_parameters, simulate


def absolute_percentage_error(measured, predicted):
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    valid = np.isfinite(measured) & np.isfinite(predicted) & (measured != 0)
    return np.divide(100*np.abs(predicted-measured), np.abs(measured),
                     out=np.full(measured.shape, np.nan), where=valid)


def response_error(measured, predicted, *, normalize=False):
    if not np.size(measured):
        return np.nan
    measured = series(measured, "Measured response")
    valid = (measured[0] >= predicted[0, 0]) & (measured[0] <= predicted[0, -1])
    actual = measured[1, valid].copy()
    values = np.interp(measured[0, valid], *predicted)
    if normalize:
        if not actual.size or actual.max() <= 0 or predicted[1].max() <= 0:
            return np.nan
        actual /= actual.max()
        values /= predicted[1].max()
    errors = absolute_percentage_error(actual, values)
    return float(np.mean(errors[np.isfinite(errors)])) if np.isfinite(errors).any() else np.nan


def evaluate_model(case, model_index, *, cancelled=None, progress=None):
    parameters = model_parameters(case["ModelParameter"][model_index])
    settings = case.get("pbmsetting", {})
    records = []
    for index, (experiment, process) in enumerate(zip(case["ExpData"], case["ProcInf"])):
        if cancelled is not None and cancelled():
            raise InterruptedError("Model evaluation stopped.")
        if progress is not None:
            progress(f"Simulating experiment {index+1}/{len(case['ExpData'])}…")
        started = monotonic()
        record = {"experiment_index": index}
        try:
            count, concentration, csd = simulate(parameters, experiment, process, case["Solubility"],
                                                 settings, cancelled=cancelled)
            seconds = monotonic()-started
            _, predicted = volume_distribution(csd, process, settings)
            predicted = np.full(3, np.nan) if predicted is None else predicted
            _, observed_traces = observed_percentiles(experiment, count[0, -1])
            observed = np.array([values[-1] if len(values) else np.nan for values in observed_traces])
            record.update(count=count, concentration=concentration, csd=csd,
                          observed_count=np.asarray(experiment.get("CountsData", [])),
                          observed_concentration=np.asarray(experiment.get("ConcentrationData", [])),
                          predicted_percentiles=predicted, observed_percentiles=observed,
                          size_errors=absolute_percentage_error(observed, predicted),
                          concentration_mape=response_error(experiment.get("ConcentrationData", []), concentration),
                          count_shape_mape=response_error(experiment.get("CountsData", []), count, normalize=True),
                          seconds=seconds, error="Success")
        except TRIAL_ERRORS as error:
            record.update(error=str(error), seconds=monotonic()-started)
        records.append(record)
        if progress is not None:
            progress(f"Experiment {index+1} · PBM {record['seconds']:.3f} s · {record['error']}")
    if not any(record["error"] == "Success" for record in records):
        raise ValueError(f"All experiment simulations failed. {records[0]['error']}")
    return {"model_index": model_index, "records": records}
