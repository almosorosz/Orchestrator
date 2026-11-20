"""Validation PBM telemetry and IMED import for explicitly marked simulated DoEs."""
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .campaign_import import CHANNELS, campaign_runs, import_run, read_json
from .distribution import volume_distribution
from .solver import simulate
from .validation.cases import Case


VERSION = "pbm-validation-v1"
PROFILE_VERSION = "pbm-calibrated-v1"
MONITOR_TYPES = {role: "mettler" if role == "AS scale" else "opcua"
                 for role in CHANNELS if role != "Temperature"}
CONTROL_TYPES = {"T": "huber", "AS": "ismatec", "seed": "masterflex",
                 "clean_sol": "ismatec", "valve_state": "nidaq_valve"}


def legacy_simulation(spec):
    return spec if spec and spec.get("version") == VERSION else None


def calibrated_simulation(options, roles, configs, selected_monitors, experiment):
    """Connect simulated PBM telemetry to the same staged commands as a real run."""
    monitors = options["monitors"]
    if not set(monitors.values()).issubset(selected_monitors):
        raise ValueError("Connect every assigned PBM monitor.")
    if any(configs.get(key, {}).get("type") != MONITOR_TYPES[role] for role, key in monitors.items()):
        raise ValueError("Use the required simulated device type for every PBM monitor.")
    controls = {role: roles[role] for role in ("T", "AS") if role in roles}
    active = {profile.device_id for profile in experiment.profiles}
    if "T" not in controls or not set(controls.values()).issubset(active):
        raise ValueError("Include the assigned temperature and antisolvent profiles in the simulated experiment.")
    if "AS" in controls and not configs[controls["AS"]].get("pump_calibration"):
        raise ValueError("Load an antisolvent pump calibration for staged PBM simulation.")
    if not all(configs.get(key, {}).get("simulated") for key in active | set(monitors.values())):
        raise ValueError("PBM simulation requires all assigned devices to be simulated.")
    start, end = experiment.model_window
    return {"version": PROFILE_VERSION, "start_s": start, "end_s": end,
            "controls": controls, "monitors": dict(monitors)}


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


def validation_model():
    parameters, _, _, _, _ = model_inputs(True)
    table = np.column_stack((np.zeros(len(parameters)), parameters, parameters, parameters))
    for index in (0, 12):
        nominal = parameters[index]
        table[index] = (1, nominal - 0.08, nominal - 0.35, nominal + 0.35)
    return {"Name": "Validation PBM", "paraset": table}


def recorded_columns(spec):
    return {
        "Temperature": f"{spec['controls']['T']}.temp_external",
        **{role: f"{key}.{'weight_g' if role == 'AS scale' else 'value'}"
           for role, key in spec["monitors"].items()},
    }


class SimulationTrace:
    def __init__(self, spec, frame, cancelled=None, *, process=None, configs=None):
        staged = spec.get("version") == PROFILE_VERSION
        if not staged:
            check_window(spec)
        self.spec = spec
        self.start, self.end = spec["start_s"], spec["end_s"]
        if staged:
            from experiment.material_balance import pump_rate_g_s
            from .campaign_import import seed_distribution

            if process is None:
                raise ValueError("Start the staged PBM simulation through the automatic loop to resolve the process inputs.")
            parameters, experiment, _, solubility, settings = model_inputs(bool(process["SeedMass"]))
            timeline = frame["time_s"].to_numpy(dtype=float)
            absolute = np.r_[self.start, timeline[(timeline > self.start) & (timeline < self.end)], self.end]
            time = absolute - self.start
            temperature = np.interp(absolute, timeline, frame[spec["controls"]["T"]])
            self.temperature = np.vstack((time, temperature))
            rate = np.zeros(len(time))
            if device := spec["controls"].get("AS"):
                rpm = np.interp(absolute, timeline, frame[device])
                rate = pump_rate_g_s(configs[device], rpm, process["AntiSolventDensity"])
            self.feed = np.vstack((time, rate))
            concentration = process["InitialConcentration"]
            experiment["ConcentrationData"] = [[0., time[-1]], [concentration, concentration]]
            if process["SeedMass"]:
                experiment["InitialCSDsData"] = seed_distribution(process["SeedPercentiles"])
        else:
            parameters, experiment, process, solubility, settings = model_inputs(spec["seeded"])
            active = frame.loc[frame["time_s"] <= self.end + 1e-8]
            time = active["time_s"].to_numpy()
            self.temperature = np.vstack((time, active[spec["controls"]["T"]]))
            # The legacy fixture uses virtual RPM values equal to synthetic g/min.
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
        active = self.start <= elapsed <= self.end + 1e-8
        elapsed -= self.start
        controls = self.spec["controls"]
        if active:
            temperature = float(np.interp(elapsed, *self.temperature))
            telemetry[controls["T"]].update(temp_external=temperature, temp_internal=temperature)
            if legacy_simulation(self.spec):
                telemetry[controls["AS"]]["flow"] = float(np.interp(elapsed, *self.feed) * 60)
        time, rate = self.feed
        clipped = np.clip(elapsed, time[0], time[-1])
        interval = np.searchsorted(time, clipped, side="right") - 1
        dt = clipped - time[interval]
        local_rate = np.interp(clipped, time, rate)
        telemetry[self.spec["monitors"]["AS scale"]]["weight_g"] = float(
            self.reservoir[interval] - dt * (rate[interval] + local_rate) / 2)
        for role, trace in self.traces.items():
            value = np.interp(elapsed, self.time, trace) if active else np.nan
            telemetry[self.spec["monitors"][role]]["value"] = float(value) if np.isfinite(value) else None


def import_simulated_campaign(folder, *, final_size_samples=1):
    """Return a complete IMED case when the recorder carries demo assignments."""
    definition_path = Path(folder) / "definition.json"
    if not definition_path.is_file():
        return None
    definitions = read_json(definition_path).get("campaign", {}).get("experiments", [])
    specs = {item["experiment_id"]: item for item in definitions if legacy_simulation(item.get("imed_simulation"))}
    if not specs:
        return None
    experiments, processes = [], []
    for number, name, file in campaign_runs(folder):
        item = specs.get(read_json(file.with_name("run.json"))["experiment_id"])
        if item is None:
            raise ValueError("The simulated campaign is missing IMED assignments for a completed run.")
        spec = item["imed_simulation"]
        check_window(spec)
        tubing = item.get("pump_adjustments") or {}
        shift = tubing.get("lead_in_s", 0.)
        _, seed_data, process, _, _ = model_inputs(spec["seeded"])
        try:
            experiment = import_run(
                file, recorded_columns(spec), start=spec["start_s"] + shift, end=spec["end_s"] + shift + 1e-8,
                concentration_multiplier=1, scale_direction=-1, smoothing=1,
                initial_concentration=None, seed_percentiles=None, no_feed=False,
                feed_channel=f"{spec['controls']['AS']}.flow", feed_multiplier=1 / 60,
                pump_adjustment=tubing.get("pumps", {}).get(spec["controls"]["AS"]),
                final_size_samples=final_size_samples,
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
    _, _, _, solubility, settings = model_inputs(True)
    return {"ExpData": experiments, "ProcInf": processes, "Solubility": solubility,
            "ModelParameter": [validation_model()],
            "pbmsetting": {**settings, "Numerics": "conservative"},
            "objweightfactor": {"W1": 0., "W2": 1., "W3": 0., "Terms": "D10 D50 D90"}}
