"""Validation PBM telemetry and IMED import for explicitly marked simulated DoEs."""
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .campaign_import import CHANNELS, campaign_runs, import_run, read_json
from .distribution import volume_distribution
from .solver import simulate
from .validation.cases import Case


VERSION = "pbm-validation-v1"
MONITOR_TYPES = {role: "mettler" if role == "AS scale" else "opcua"
                 for role in CHANNELS if role != "Temperature"}
CONTROL_TYPES = {"T": "huber", "AS": "ismatec", "seed": "masterflex",
                 "clean_sol": "ismatec", "valve_state": "nidaq_valve"}


def compile_simulation(options, controls, configs, selected_monitors, monitors, profiles):
    check_window(options)
    monitors = options.get("monitors", {}) if monitors is None else monitors
    if set(monitors) != set(MONITOR_TYPES):
        raise ValueError("Assign each IMED recorded channel before previewing.")
    assignments = [(controls.get(role), kind) for role, kind in CONTROL_TYPES.items()]
    assignments += [(monitors[role], kind) for role, kind in MONITOR_TYPES.items()]
    ids = [device for device, _ in assignments]
    if len(set(ids)) != len(ids) or not set(monitors.values()).issubset(selected_monitors):
        raise ValueError("Use distinct devices and connect every assigned IMED monitor.")
    if any(configs.get(device, {}).get("type") != kind for device, kind in assignments):
        raise ValueError("Use the required simulated device type for every IMED assignment.")
    if any(not configs[key].get("simulated") for key in set(controls.values()) | set(selected_monitors)):
        raise ValueError("The IMED validation campaign requires all assigned devices to be simulated.")
    if any(configs[controls[role]].get("pump_calibration") for role in ("AS", "clean_sol")):
        raise ValueError("Use the default 1 g/mL simulated pump conversion for AS and cleaning.")
    spec = {**options, "controls": controls, "monitors": monitors}
    by_role = {role: next(p for p in profiles if p.device_id == controls[role])
               for role in CONTROL_TYPES}
    end, hold = spec["end_s"], spec["hold_start_s"]
    temperature = by_role["T"].points
    final = next((p.target for p in temperature if np.isclose(p.time_s, end)), None)
    if (final is None or not any(np.isclose(p.time_s, hold) for p in temperature)
            or any(p.target != final for p in temperature
                   if hold - 1e-8 <= p.time_s <= end + 1e-8)):
        raise ValueError("The final temperature must be held through the equilibration window.")
    if any(p.target < 0 or (p.time_s >= hold and p.target != 0) for p in by_role["AS"].points):
        raise ValueError("End antisolvent feed before final equilibration.")
    seed = by_role["seed"].points
    if (bool(seed[0].target > 0) != spec["seeded"]
            or (not spec["seeded"] and any(p.target for p in seed))
            or any(p.target < 0 or (p.time_s >= hold and p.target != 0) for p in seed)):
        raise ValueError("Seeded profiles must start with the seed pulse at time zero.")
    for role in ("valve_state", "clean_sol"):
        points = by_role[role].points
        if any(p.target for p in points if p.time_s <= end) or not any(p.target > 0 for p in points):
            raise ValueError("Schedule discharge and cleaning after the crystallization window.")
    return spec


def check_window(spec):
    if not isinstance(spec, dict):
        raise ValueError("Invalid simulated IMED process window.")
    hold, end = spec.get("hold_start_s"), spec.get("end_s")
    if (spec.get("version") != VERSION or not isinstance(spec.get("seeded"), bool)
            or spec.get("start_s") != 0 or not isinstance(hold, (float, int))
            or not isinstance(end, (float, int)) or not 0 < hold < end < np.inf):
        raise ValueError("Invalid simulated IMED process window.")


def model_inputs(seeded):
    parameters, experiment, process, solubility, settings = Case("combined", "nucleation").inputs(48)
    if not seeded:
        process["SeedMass"] = 0.
        experiment["InitialCSDsData"] = np.empty((2, 0))
    return parameters, experiment, process, solubility, settings


def recorded_columns(spec):
    return {
        "Temperature": f"{spec['controls']['T']}.temp_external",
        **{role: f"{key}.{'weight_g' if role == 'AS scale' else 'value'}"
           for role, key in spec["monitors"].items()},
    }


class SimulationTrace:
    def __init__(self, spec, frame, cancelled=None):
        check_window(spec)
        self.spec = spec
        self.end = spec["end_s"]
        parameters, experiment, process, solubility, settings = model_inputs(spec["seeded"])
        active = frame.loc[frame["time_s"] <= self.end + 1e-8]
        time = active["time_s"].to_numpy()
        self.temperature = np.vstack((time, active[spec["controls"]["T"]]))
        # The assigned Ismatec uses mL/min and simulated AS density is 1 g/mL.
        self.feed = np.vstack((time, active[spec["controls"]["AS"]] / 60))
        experiment.update(TemperatureData=self.temperature, FlowRateData=self.feed)
        count, concentration, csd, state = simulate(
            parameters, experiment, process, solubility, settings,
            return_history=True, cancelled=cancelled)
        self.time = count[0]
        self.traces = {"Concentration": concentration[1], "FBRM": count[1]}
        sizes = np.full((3, len(self.time)), np.nan)
        for index, density in enumerate(state[:len(csd[0])].T):
            if cancelled is not None and cancelled():
                raise InterruptedError("Simulated IMED experiment stopped.")
            _, percentiles = volume_distribution((csd[0], density), process, settings)
            if percentiles is not None:
                sizes[:, index] = percentiles
        self.traces.update(zip(("D10", "D50", "D90"), sizes))
        self.reservoir = 200 - cumulative_trapezoid(self.feed[1], time, initial=0)

    def apply(self, telemetry, elapsed):
        active = elapsed <= self.end + 1e-8
        controls = self.spec["controls"]
        if active:
            temperature = float(np.interp(elapsed, *self.temperature))
            telemetry[controls["T"]].update(temp_external=temperature, temp_internal=temperature)
            telemetry[controls["AS"]]["flow"] = float(np.interp(elapsed, *self.feed) * 60)
        time, rate = self.feed
        clipped = min(elapsed, time[-1])
        interval = np.searchsorted(time, clipped, side="right") - 1
        dt = clipped - time[interval]
        local_rate = np.interp(clipped, time, rate)
        telemetry[self.spec["monitors"]["AS scale"]]["weight_g"] = float(
            self.reservoir[interval] - dt * (rate[interval] + local_rate) / 2)
        for role, trace in self.traces.items():
            value = np.interp(elapsed, self.time, trace) if active else np.nan
            telemetry[self.spec["monitors"][role]]["value"] = float(value) if np.isfinite(value) else None


def import_simulated_campaign(folder):
    """Return a complete IMED case when the recorder carries demo assignments."""
    definition_path = Path(folder) / "definition.json"
    if not definition_path.is_file():
        return None
    definitions = read_json(definition_path).get("campaign", {}).get("experiments", [])
    specs = {item["experiment_id"]: item for item in definitions if item.get("imed_simulation")}
    if not specs:
        return None
    experiments, processes = [], []
    for number, name, file in campaign_runs(folder):
        item = specs.get(read_json(file.with_name("run.json"))["experiment_id"])
        if item is None:
            raise ValueError("The simulated campaign is missing IMED assignments for a completed run.")
        spec = item["imed_simulation"]
        check_window(spec)
        _, seed_data, process, _, _ = model_inputs(spec["seeded"])
        try:
            experiment = import_run(
                file, recorded_columns(spec), start=spec["start_s"], end=spec["end_s"] + 1e-8,
                concentration_multiplier=1, scale_direction=-1, smoothing=1,
                initial_concentration=None, seed_percentiles=None, no_feed=False,
                feed_channel=f"{spec['controls']['AS']}.flow", feed_multiplier=1 / 60,
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"{item['name']}: {error}") from error
        if not np.isclose(experiment["TemperatureData"][0, -1], spec["end_s"], atol=1e-8, rtol=0):
            raise ValueError(f"{item['name']}: recording does not cover the end of equilibration.")
        size_traces = [experiment[f"{role}Data"] for role in ("D10", "D50", "D90")]
        if (experiment["CountsData"].shape[1] < 2 or experiment["FinalD10D50D90"].size != 3
                or any(trace.shape[1] < 2 for trace in size_traces)):
            raise ValueError(f"{item['name']}: assigned counts or size channels are missing samples.")
        if spec["seeded"] and any(trace[0, 0] != 0 for trace in size_traces):
            raise ValueError(f"{item['name']}: seeded run is missing the initial size readings.")
        experiment.update(Name=spec.get("label", item["name"]), RecordingFile=str(file),
                          InitialCSDsData=np.asarray(seed_data["InitialCSDsData"]),
                          FinalEquilibrationStart_s=spec["hold_start_s"])
        if spec["seeded"]:
            experiment["InitialD10D50D90"] = np.array([trace[1, 0] for trace in size_traces])
        experiments.append(experiment)
        processes.append({**process, "No": number or len(processes) + 1})
    parameters, _, _, solubility, settings = model_inputs(True)
    table = np.column_stack((np.zeros(len(parameters)), parameters, parameters, parameters))
    for index in (0, 12):
        nominal = parameters[index]
        table[index] = (1, nominal - 0.08, nominal - 0.35, nominal + 0.35)
    return {"ExpData": experiments, "ProcInf": processes, "Solubility": solubility,
            "ModelParameter": [{"Name": "Validation PBM", "paraset": table}],
            "pbmsetting": {**settings, "Numerics": "conservative"},
            "objweightfactor": {"W1": 0., "W2": 1., "W3": 0., "Terms": "D10 D50 D90"}}
