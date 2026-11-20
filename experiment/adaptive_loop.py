"""Prepare, model and compile the next campaign for a dashboard-driven loop."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
from importlib.util import find_spec
from pathlib import Path
import csv
from tempfile import TemporaryDirectory

import numpy as np

from bo.profiles import Profile, case_runs, make_profile, recipe_data, campaign_profiles
from experiment.doe_decisions import (
    capture_decisions, charge_volume, decision_summary, planned_decisions, starting_run_process, validate_decisions,
)
from experiment.doe_campaign import ROLES, convert_profile
from experiment import loop_outputs
from experiment.pump_calibration_files import seed_dose_command, seed_dose_speed
from experiment.loop_process import (
    PROCESS_FIELDS, VALIDATION_PE_SETTINGS, require_positive_process, starting_process_defaults,
)
from experiment.models import Campaign, DeviceProfile, ExperimentDefinition, ProfilePoint
from experiment.recording import atomic_json
from hardware_devices.capabilities import is_monitor_only
from hardware_devices.limits import target_allowed, target_limits
from imed_python.campaign_import import ALIASES, CHANNELS, campaign_runs, import_run, read_json, seed_distribution
from imed_python.demo_campaign import calibrated_simulation, compile_simulation, import_simulated_campaign, legacy_simulation, model_inputs, validation_model
from imed_python.design import discrimination_score, d_optimal_log_score, optimize_recipe, parameter_samples, recipe_space
from imed_python.estimation import FitObjective, fit_model
from imed_python.model_screening import matlab_screening_fields, pack_screening, screen_models
from imed_python.native_case import save_native_case
from imed_python.pe_settings import PE_WEIGHTS_KEY, PYFIT_DEFAULTS, PYFIT_SETTINGS_KEY, effective_pe_pbm
from imed_python.session import display_name, loop_stages, pack_fit, pe_signatures, record_step, signature
from imed_python.solver import pbm_backend_label
from imed_python.size_data import validate_final_size_samples
from imed_python.mixing import DEFAULT_MIXING, validate_mixing, average_mixing, trace_mean


MODES = (("External MODEL file", "external"), ("Bayesian optimization", "bo"), ("IMED", "imed"))
POSITIVE_PROCESS = ("SolvMass", "SolventDensity", "AntiSolventDensity", "SolidDensity", "ShapeFactor")
NONNEGATIVE_PROCESS = ("AntiSolvMass", "SeedMass", "PowerRatio")
DEVICE_RECORDING_COLUMNS = {
    "huber": ("temp_external", "temp_internal"), "opcua": ("value",), "mettler": ("weight_g",),
    "jkem": ("speed", "target"), "ismatec": ("speed_rpm", "flow", "target"), "masterflex": ("speed", "target"),
}
IMED_OBJECTIVES = (("FBRM", ("FBRM",)), ("Concentration", ("Concentration",)),
                   ("D10 / D50 / D90", ("D10", "D50", "D90")))


def default_setup():
    return {
        "max_followups": 1, "channels": {}, "process": {}, "initial_processes": {},
        "recording": {"start_s": 0, "end_s": None, "concentration_multiplier": 1,
                      "initial_concentration": None, "scale_direction": -1, "smoothing": 5,
                      "seed_percentiles": [], "no_feed": False},
        "manual_assignments": {}, "manual_simulation": None, "followup_start_s": 0,
        "experiment_period": {"mode": "definition", "stage_ids": []},
        "reactor_unit": "", "mixing": dict(DEFAULT_MIXING),
        "derive_initial_charges": True,
        "doe_decisions": None,
        "support_profiles": {}, "seed_pump": None,
        "imed": {"delta_aic": 2, "iterations": 5, "covariance_samples": 8,
                 "equilibration_s": 5, "conditions": {}, "use_doe_decisions": True},
    }


def _number(value, label, low=0, high=np.inf):
    if (not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool)
            or not np.isfinite(value) or not low <= value <= high):
        raise ValueError(f"Invalid {label}.")
    return value


def _integer(value, label, low, high):
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer.")
    return _number(value, label, low, high)


def imed_response_weights(case):
    stored = case.get(PE_WEIGHTS_KEY, case.get("objweightfactor", {}))
    stored = stored if isinstance(stored, dict) else {}
    return [float(_number(stored.get(f"W{i}", 0), f"{label} objective weight", 0, 10000))
            for i, (label, _) in enumerate(IMED_OBJECTIVES, 1)]


def imed_objective_channels(weights):
    return tuple(channel for (_, channels), weight in zip(IMED_OBJECTIVES, weights)
                 if weight > 0 for channel in channels)


def imed_design_conditions(conditions, *, no_feed):
    if not conditions or not no_feed:
        return conditions
    solvent, antisolvent = conditions["liquid_masses"]
    return {**conditions, "no_feed": True, "final_anti": antisolvent / (solvent + antisolvent) if solvent > 0 else 0.,
            "antisolvent_rates": (0., 0.)}


def _merge(defaults, updates, label):
    if not isinstance(updates, dict):
        raise ValueError(f"{label.capitalize()} must contain named settings.")
    unknown = set(updates) - set(defaults)
    if unknown:
        fields = ", ".join(sorted(map(str, unknown)))
        hint = " Open Automatic loop setup to review and save the supported settings." if label == "automatic loop settings" else ""
        raise ValueError(f"Unsupported fields in {label}: {fields}.{hint}")
    return {**defaults, **updates}


def setup_for_editing(updates):
    """Open supported settings from a saved setup without applying unknown fields."""
    if not isinstance(updates, dict):
        return validate_setup(updates), []
    keys = default_setup().keys()
    supported = {key: value for key, value in updates.items() if key in keys}
    # A file with no loop settings may be a different configuration entirely.
    normalized = validate_setup(supported if supported else updates)
    return normalized, sorted(map(str, set(updates) - keys))


def validate_setup(updates):
    defaults = default_setup()
    data = _merge(defaults, updates, "automatic loop settings")
    if not isinstance(data["reactor_unit"], str):
        raise ValueError("Select a model reactor unit.")
    if type(data["derive_initial_charges"]) is not bool:
        raise ValueError("Choose whether to derive starting charges from preparation pumps.")
    data["mixing"] = validate_mixing(data["mixing"])
    data["doe_decisions"] = validate_decisions(data["doe_decisions"])
    period = data["experiment_period"] = _merge(defaults["experiment_period"], data["experiment_period"], "experiment period")
    if (period["mode"] not in {"definition", "stages", "manual"} or not isinstance(period["stage_ids"], list)
            or any(not isinstance(key, str) or not key for key in period["stage_ids"])
            or len(set(period["stage_ids"])) != len(period["stage_ids"])):
        raise ValueError("Choose a valid experiment period and distinct stage IDs.")
    if period["mode"] == "stages" and not period["stage_ids"]:
        raise ValueError("Select at least one stage for the experiment period.")
    _integer(data["max_followups"], "maximum follow-up experiments", 1, 1000)
    _number(data["followup_start_s"], "follow-up crystallization start")
    manual = data["manual_assignments"]
    if (not isinstance(manual, dict) or set(manual) - set(ROLES)
            or any(not isinstance(device, str) or not device for device in manual.values())
            or len(set(manual.values())) != len(manual)):
        raise ValueError("Map each manual profile role to a distinct control device.")
    if data["manual_simulation"] is not None:
        from imed_python.demo_campaign import MONITOR_TYPES

        simulation = _merge({"hold_start_s": None, "end_s": None, "monitors": {}},
                            data["manual_simulation"], "manual PBM simulation")
        hold = _number(simulation["hold_start_s"], "simulated equilibration start", 1e-12)
        if _number(simulation["end_s"], "simulated recording end") <= hold:
            raise ValueError("Simulation end must follow the equilibration start.")
        monitors = simulation["monitors"]
        if (not isinstance(monitors, dict) or set(monitors) != set(MONITOR_TYPES)
                or any(not isinstance(device, str) or not device for device in monitors.values())
                or len(set(monitors.values())) != len(monitors)):
            raise ValueError("Assign a distinct simulated device for each PBM telemetry channel.")
        data["manual_simulation"] = simulation
    mapping = data["channels"]
    if (not isinstance(mapping, dict) or set(mapping) - set(CHANNELS)
            or any(not isinstance(value, str) for value in mapping.values())):
        raise ValueError("Measurement mappings must name execution.csv columns for known IMED channels.")
    for key in ("process", "initial_processes", "support_profiles"):
        if not isinstance(data[key], dict):
            raise ValueError(f"{key} must contain named entries.")
    if any(not isinstance(value, dict) for value in data["initial_processes"].values()):
        raise ValueError("Initial process overrides must be grouped by experiment name.")
    recording = data["recording"] = _merge(defaults["recording"], data["recording"], "recording")
    start = _number(recording["start_s"], "recording start")
    if recording["end_s"] is not None and _number(recording["end_s"], "recording end") <= start:
        raise ValueError("Recording end must follow the start.")
    _number(recording["concentration_multiplier"], "concentration multiplier", 1e-12)
    if recording["initial_concentration"] is not None:
        _number(recording["initial_concentration"], "initial concentration", 0, 1 - 1e-12)
    if (type(recording["scale_direction"]) is not int or recording["scale_direction"] not in (-1, 1)
            or type(recording["no_feed"]) is not bool):
        raise ValueError("Scale direction must be −1 or 1 and no_feed must be true or false.")
    _integer(recording["smoothing"], "scale median window", 1, 101)
    seed = recording["seed_percentiles"]
    if not isinstance(seed, list) or (seed and len(seed) != 3):
        raise ValueError("Enter three seed percentiles in µm or an empty list.")
    if seed:
        seed_distribution([_number(value, "seed percentile") for value in seed])
    if data["seed_pump"] is not None:
        pump = _merge({"rpm": 0, "grams_per_second": None, "use_device_calibration": False}, data["seed_pump"], "seed pump")
        _number(pump["rpm"], "seed pump RPM", 1e-12)
        if type(pump["use_device_calibration"]) is not bool:
            raise ValueError("Choose whether to use the seed pump's loaded calibration.")
        if not pump["use_device_calibration"]:
            _number(pump["grams_per_second"], "seed pump mass rate", 1e-12)
    imed = data["imed"] = _merge(defaults["imed"], data["imed"], "IMED design")
    if type(imed["use_doe_decisions"]) is not bool:
        raise ValueError("Choose whether IMED uses the DoE decision variables.")
    _number(imed["delta_aic"], "relative AIC threshold")
    _number(imed["equilibration_s"], "final equilibration duration")
    _integer(imed["iterations"], "design iterations", 0, 1000)
    _integer(imed["covariance_samples"], "covariance samples", 2, 10000)
    names = ("batch", "temperature", "temperature_rates", "liquid_masses", "concentration",
             "final_anti", "antisolvent_rates", "seed_bounds")
    conditions = _merge(dict.fromkeys(names), imed["conditions"], "IMED design conditions")
    if imed["conditions"]:
        for field in ("batch", "concentration", "final_anti"):
            _number(conditions[field], field)
        for field in set(names) - {"batch", "concentration", "final_anti"}:
            values = conditions[field]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"Enter two values for {field}.")
            for value in values:
                _number(value, field, -273.149 if field == "temperature" else 0)
        recipe_space(imed_design_conditions(conditions, no_feed=recording["no_feed"]))
    return data


def check_loop_folders(experiment_path, model_path):
    problems = []
    for label, path in (("Experiment", experiment_path), ("MODEL", model_path)):
        if not path or not Path(path).is_dir():
            problems.append(f"Choose an existing {label} folder.")
            continue
        try:
            prefix = ".crystaloop-" + ("0" * 30 if label == "Experiment" else "")
            with TemporaryDirectory(dir=path, prefix=prefix) as directory:
                probe = Path(directory)
                if label == "Experiment":
                    probe /= "0001_" + "0" * 32
                    probe.mkdir()
                (probe / "samples.pending.jsonl").write_text("CrystaLoop", encoding="utf-8")
        except OSError as error:
            problems.append(f"Cannot write campaign files under the {label} folder: {error}")
    return problems


def _process(values):
    require_positive_process(values, POSITIVE_PROCESS)
    for key in NONNEGATIVE_PROCESS:
        _number(values.get(key), PROCESS_FIELDS[key])
    return values


def _role_active(campaign, device_id, *, modeling_only=False, recording=None):
    if campaign is None:
        return False
    for exp in campaign.experiments:
        for profile in exp.profiles:
            if profile.device_id != device_id:
                continue
            window = exp.model_window
            if window is None and recording is not None:
                window = (recording["start_s"], recording["end_s"] or
                          max(p.points[-1].time_s for p in exp.profiles))
            if not modeling_only or window is None:
                if any(point.target != 0 for point in profile.points):
                    return True
                continue
            begin, end = window
            if end <= begin:
                continue
            times = np.array([point.time_s for point in profile.points])
            targets = np.array([point.target for point in profile.points])
            sample_times = np.union1d([begin, np.nextafter(end, begin)], times[(times >= begin) & (times < end)])
            values = (targets[np.clip(np.searchsorted(times, sample_times, side="right") - 1, 0, len(targets) - 1)]
                      if profile.interpolation == "step" else np.interp(sample_times, times, targets))
            if np.any(values != 0):
                return True
    return False


def device_configuration_changed(previous, current):
    fields = ("type", "connection", "defaults", "simulated", "target_limits", "pump_calibration", "connectivity",
              "pump_corrections", "seed_calibration", "reservoir_density_g_ml")
    return previous is None or current is None or any(current.get(field) != previous.get(field) for field in fields)


def _column_exists(column, configs):
    device, _, field = column.rpartition(".")
    kind = configs.get(device, {}).get("type")
    return field in DEVICE_RECORDING_COLUMNS.get(kind, ())


def recording_mapping(setup, roles, configs, *, required_measurements=None):
    mapping = {key: next((column for column in ALIASES[key] if _column_exists(column, configs)), "")
               for key in CHANNELS}
    mapping["Temperature"] = f"{roles['T']}.temp_external"
    mapping.update(setup["channels"])
    if setup["recording"]["no_feed"]:
        mapping["AS scale"] = ""
    required = set(CHANNELS if required_measurements is None else required_measurements) | {"Temperature"}
    invalid = []
    for measurement, column in mapping.items():
        if column and not _column_exists(column, configs):
            if measurement in required:
                invalid.append(f"{measurement} ({column})")
            else:
                mapping[measurement] = ""
    if invalid:
        raise ValueError("In Recording, select assigned devices or monitors for " + ", ".join(invalid) + ".")
    columns = [column for column in mapping.values() if column]
    if len(set(columns)) != len(columns):
        raise ValueError("Map each recorded measurement to a distinct column.")
    return mapping


def antisolvent_recording_source(recording, mapping, campaign, roles, configs, *, infer_no_feed=False):
    as_roles = {"AS"} | {role for role, lane in (campaign.line_roles if campaign else {}).items()
                         if lane.get("role") == "AS"}
    pumps = {roles[role] for role in as_roles if role in roles}
    active = any(_role_active(campaign, pump, modeling_only=True, recording=recording) for pump in pumps)
    if recording["no_feed"]:
        if active:
            raise ValueError("No antisolvent feed is selected, but an AS pump runs during the experiment period. "
                             "Remove that delivery or clear the no-feed checkbox in Recording.")
        return "No antisolvent feed"
    if mapping.get("AS scale"):
        return f"AS scale ({mapping['AS scale']})"
    pump = configs.get(roles.get("AS") or (next(iter(pumps)) if len(pumps) == 1 else None), {})
    if pump.get("pump_calibration"):
        return "Calibrated AS pump delivery"
    if pump.get("type") == "ismatec" and pump.get("simulated"):
        return "Simulated AS pump delivery"
    if infer_no_feed and not active:
        return "No antisolvent feed"
    raise ValueError("Map the AS scale or load the assigned antisolvent pump's mass-flow calibration.")


def read_completed_recordings(folder, expected, recording, mapping, processes_by_id, windows, *,
                              require_initial_concentration=True, roles=None, mixing=None, allow_density_fallback=False,
                              final_size_samples=1):
    runs = campaign_runs(folder)
    run_ids = [read_json(file.with_name("run.json"))["experiment_id"] for _, _, file in runs]
    if set(run_ids) != set(expected) or len(set(run_ids)) != len(runs):
        raise ValueError("Recorded experiment identities differ from the completed campaign manifest.")
    demo = import_simulated_campaign(folder, final_size_samples=final_size_samples)
    if demo is not None:
        return demo["ExpData"], demo["ProcInf"], runs
    experiments, processes = [], []
    definition_file = Path(folder) / "definition.json"
    payload = read_json(definition_file) if definition_file.is_file() else {}
    definitions = payload.get("campaign", {}).get("experiments", [])
    definitions = {item["experiment_id"]: item for item in definitions}
    configs = payload.get("devices", {})
    roles = roles or {key: lane["device_id"] for key, lane in payload.get("campaign", {}).get("line_roles", {}).items()}
    for (number, name, file), experiment_id in zip(runs, run_ids):
        process = processes_by_id[experiment_id]
        definition = definitions.get(experiment_id, {})
        tubing = definition.get("pump_adjustments") or {}
        start, end = windows.get(experiment_id, (recording["start_s"], recording["end_s"]))
        if tubing:
            shift = tubing["lead_in_s"]
            start += shift
            end = tubing["nominal_end_s"] if end is None else min(end + shift, tubing["nominal_end_s"])
        initial = process.get("InitialConcentration", recording["initial_concentration"])
        with file.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        headers = rows[0].keys() if rows else ()
        def series(column):
            points = {}
            for row in rows:
                try:
                    time, value = float(row["elapsed_s"]), float(row.get(column, ""))
                except (ValueError, TypeError):
                    continue
                if (np.isfinite(time) and np.isfinite(value) and time >= start and (end is None or time <= end)
                        and str(row.get(column.rpartition('.')[0] + ".connected", "true")).lower() not in {"false", "0"}):
                    points[time] = value
            return np.asarray(sorted(points.items()), dtype=float).reshape(-1, 2).T
        feed_options = {}
        source = "No antisolvent feed" if recording["no_feed"] else "AS scale"
        if not recording["no_feed"] and series(mapping.get("AS scale", "")).shape[1] < 2:
            device = roles.get("AS")
            cfg = configs.get(device, {})
            fields = ("flow", "speed_rpm", "speed", "target")
            if cfg.get("type") == "ismatec":
                fields = ("speed_rpm", "target", "flow")
            if cfg.get("type") == "masterflex" and cfg.get("defaults", {}).get("direction") == "-":
                fields = ("target", "speed_rpm", "speed")
            field = next((field for field in fields
                          if f"{device}.{field}" in headers and series(f"{device}.{field}").shape[1] >= 2), None)
            if not device or not field:
                raise ValueError("No usable AS scale. Assign an antisolvent pump with recorded delivery and a calibration/density.")
            source = f"Calibrated pump delivery ({device}.{field})"
            if cfg.get("pump_calibration"):
                from experiment.pump_calibration_files import delivered_flow
                transform = lambda values: delivered_flow(cfg, values) / 3600.
            elif cfg.get("type") == "ismatec" and cfg.get("simulated") and allow_density_fallback:
                density = _number(process.get("AntiSolventDensity"), "antisolvent density", 1e-12)
                transform = lambda values: values * density / 60.
                source = f"Estimated pump delivery from AS density ({device}.{field})"
            elif cfg.get("type") == "ismatec" and cfg.get("simulated"):
                transform = lambda values: values / 60.
                source = f"Simulated pump delivery ({device}.{field})"
            else:
                raise ValueError(f"Load a mass-flow calibration for antisolvent pump {device}, or map its reservoir scale.")
            feed_options = {"feed_channel": f"{device}.{field}", "feed_transform": transform}
        experiment = import_run(
            file, mapping, start=start, end=end, initial_concentration=initial,
            concentration_multiplier=recording["concentration_multiplier"],
            scale_direction=recording["scale_direction"], smoothing=recording["smoothing"],
            seed_percentiles=recording["seed_percentiles"] if process["SeedMass"] else None,
            no_feed=recording["no_feed"], require_initial_concentration=require_initial_concentration,
            pump_adjustment=tubing.get("pumps", {}).get(roles.get("AS")),
            final_size_samples=final_size_samples,
            **feed_options,
        )
        experiment.update(Name=name, RecordingFile=str(file))
        experiment["CommandedProfiles"] = {key: value.payload() for key, value in campaign_profiles(experiment, process, payload, roles).items()}
        experiment["AntisolventSource"] = source
        if definition.get("run_stages"):
            experiment["ModelingWindow_s"] = [start, end]
            experiment["ModelingStages"] = [stage for stage in definition["run_stages"]
                                            if start <= stage["start_s"] and (end is None or stage["end_s"] <= end)]
        duration = float(experiment["TemperatureData"][0, -1])
        origin = experiment["RecordingStart_s"]
        stirrer = roles.get("stirring_rate")
        speed_field = next((field for field in ("speed_rpm", "speed", "target")
                            if f"{stirrer}.{field}" in headers and series(f"{stirrer}.{field}").shape[1] >= 2), None)
        speed = series(f"{stirrer}.{speed_field}") if speed_field else None
        speed_step = False
        speed_source = "Assigned stirrer commands" if speed_field == "target" else "Measured stirring rate"
        if speed is not None:
            speed[0] -= origin
        else:
            entry = next((entry for entry in definition.get("profiles", []) if entry["device_id"] == stirrer), None)
            if entry:
                raw = np.asarray([[p["time_s"] for p in entry["points"]], [p["target"] for p in entry["points"]]])
                times = np.unique(np.r_[origin, raw[0, (raw[0] > origin) & (raw[0] < origin + duration)], origin + duration])
                speed_step = entry.get("interpolation") == "step"
                values = raw[1, np.clip(np.searchsorted(raw[0], times, side="right") - 1, 0, len(raw[0]) - 1)] if speed_step else np.interp(times, *raw)
                speed = np.vstack((times - origin, values))
                speed_source = "Assigned stirrer commands"
        mean_rpm = trace_mean(speed, duration, "Stirring rate", step=speed_step) if speed is not None else None
        if mean_rpm is not None:
            process = {**process, "StirringRPM": mean_rpm}
        if all(key in process for key in ("SolvMass", "AntiSolvMass", "SolventDensity", "AntiSolventDensity")):
            volume = process.get("InitialVolume_ml", process["SolvMass"] / process["SolventDensity"] + process["AntiSolvMass"] / process["AntiSolventDensity"])
            added_mean = trace_mean(experiment["AddedAntisolventData"], duration, "Added antisolvent mass")
            process = {**process, "InitialVolume_ml": volume,
                       "AverageVolume_ml": volume + added_mean / process["AntiSolventDensity"]}
        selected_mixing = mixing if mixing is not None else process.get("MixingSettings")
        if selected_mixing is not None:
            process = average_mixing(experiment, process, selected_mixing, mean_rpm=mean_rpm, source=speed_source)
        experiments.append(experiment)
        processes.append({**process, "No": number or len(processes) + 1})
    return experiments, processes, runs


def _replace_model_window(original, replacement, begin, end, new_end, *, suffix_starts_at_boundary=False, run_end=None):
    """Splice the proposed profile into the run without ramping across its phase boundaries."""
    interpolation = "step" if original.interpolation == replacement.interpolation == "step" else "linear"

    def points_for(profile, offset=0.):
        points = []
        for point in profile.points:
            time = point.time_s + offset
            if points and time <= points[-1].time_s:
                # A short edge can lose its distinct time when moved to the run clock.
                time = float(np.nextafter(points[-1].time_s, np.inf))
            if points and interpolation == "linear" and profile.interpolation == "step" and point.target != points[-1].target:
                before = time - min(1e-6, (time - points[-1].time_s) / 2)
                if points[-1].time_s < before < time:
                    points.append(ProfilePoint(before, points[-1].target))
            points.append(ProfilePoint(time, point.target))
        return points

    existing = points_for(original)
    middle = points_for(replacement, begin)
    if middle[-1].time_s < new_end:
        middle.append(ProfilePoint(new_end, middle[-1].target))
    delta = max(new_end, middle[-1].time_s) - end
    before = [p for p in existing if p.time_s < begin]
    after = [p for p in existing if p.time_s > end]
    has_suffix = bool(after) or (run_end is not None and end < run_end)
    x, y = zip(*[(p.time_s, p.target) for p in existing])
    if begin > 0:
        if not before:
            before.append(ProfilePoint(0., y[0]))
        if interpolation == "linear":
            boundary = float(np.nextafter(begin, 0.))
            if before[-1].time_s < boundary:
                before.append(ProfilePoint(boundary, float(np.interp(boundary, x, y))))
    if has_suffix and suffix_starts_at_boundary:
        # The next stage owns the setpoint at the boundary. Retain the
        # recommendation's endpoint as the left limit for the modeled period.
        last = float(np.nextafter(new_end, begin))
        endpoint = middle[-1].target
        middle = [point for point in middle if point.time_s < last]
        middle.append(ProfilePoint(last, endpoint))
        delta = new_end - end
    points = before + middle
    if has_suffix:
        value = (y[max(0, np.searchsorted(x, end, side="right") - 1)] if interpolation == "step"
                 else float(np.interp(end, x, y)))
        time = new_end if suffix_starts_at_boundary else float(np.nextafter(points[-1].time_s, np.inf))
        points.append(ProfilePoint(time, value))
    for point in after:
        time = max(point.time_s + delta, float(np.nextafter(points[-1].time_s, np.inf)))
        points.append(ProfilePoint(time, point.target))
    return replace(original, points=tuple(points), interpolation=interpolation)


def _followup_time(time, begin, end, new_end):
    if time <= begin:
        return time
    if time >= end:
        return new_end + (time - end)
    return begin + (time - begin) * (new_end - begin) / (end - begin)


class RecipeCompiler:
    """Resolve process profiles to the assigned devices for LOOP and one-off execution."""

    def __init__(self, setup, template, configs, calibration_storage, assignments, monitors, mode):
        from experiment.run_stages import modeling_campaign
        self.setup, self.template, self.mode = setup, modeling_campaign(template, setup), mode
        self.staged = bool(self.template and self.template.experiments[0].run_stages)
        self.calibration_storage = calibration_storage
        self.assignments = {item.role: item for item in assignments}
        self.line_roles = template.line_roles if template else {}
        self.role_specs = {role: ROLES[self.line_roles.get(role, {}).get("role", role)] for role in self.assignments}
        self.roles = {role: item.device_id for role, item in self.assignments.items()}
        for semantic in ("T", "AS", "seed", "stirring_rate"):
            matches = [role for role in self.roles if self.line_roles.get(role, {}).get("role", role) == semantic]
            if semantic not in self.roles and len(matches) == 1:
                original = matches[0]
                self.roles[semantic] = self.roles[original]
                self.assignments[semantic] = self.assignments[original]
                self.role_specs[semantic] = ROLES[semantic]
        self.monitors = tuple(monitors)
        self.simulation = template.experiments[0].imed_simulation if template else None
        self.demo = legacy_simulation(self.simulation)
        selected = set(self.roles.values()) | set(self.monitors)
        self.configs = deepcopy({key: configs[key] for key in selected if key in configs})
        if "T" not in self.roles:
            raise ValueError("Assign a reactor temperature controller in the DoE or Automatic loop setup → Devices.")
        for role, device in self.roles.items():
            cfg = configs.get(device, {})
            if cfg.get("type") not in self.role_specs[role].compatible_types or is_monitor_only(cfg):
                raise ValueError(f"Assign a compatible controlled device for {self.role_specs[role].label}.")

    def process_defaults(self, case):
        return starting_process_defaults(self.setup, case, self.demo)

    def check_devices(self, campaign, configs, connected, *, require_all=False):
        actuators = {profile.device_id for experiment in campaign.experiments for profile in experiment.profiles}
        used = (set(self.roles.values()) if require_all else actuators) | set(campaign.monitor_device_ids)
        unavailable = used - set(connected)
        if unavailable:
            raise ValueError("Connect assigned devices and monitors: " + ", ".join(sorted(unavailable)))
        for key in used:
            if device_configuration_changed(self.configs.get(key), configs.get(key)):
                raise ValueError(f"Device {key} changed since the recipe was assigned.")
        for key in campaign.monitor_device_ids:
            if not is_monitor_only(configs[key]):
                raise ValueError(f"Assign {key} as a read-only monitor.")
        for experiment in campaign.experiments:
            ExperimentDefinition.from_frame(experiment.to_frame(), configs)

    @property
    def seed_rpm(self):
        device = self.roles.get("seed")
        return seed_dose_speed(self.configs[device], self.setup["seed_pump"]) if device else None

    def check_support(self, campaign):
        support = self.setup["support_profiles"]
        bad_roles = set(support) - (set(self.roles) - {"T", "AS"})
        if bad_roles:
            raise ValueError("Support profiles need distinct assigned DoE roles: " + ", ".join(sorted(bad_roles)))
        for role, profile in support.items():
            self._support_profile(role, profile)
        if self.setup["seed_pump"] and "seed" not in self.roles:
            raise ValueError("Assign the seed pump before enabling calibrated seed delivery.")
        if self.seed_rpm is not None:
            cfg = self.configs[self.roles["seed"]]
            if not target_allowed(self.seed_rpm, target_limits(cfg)):
                raise ValueError("Calibrated seed pump RPM exceeds the assigned pump limits.")
            seed_dose_command(cfg, self.setup["seed_pump"], 0.)
        if self.line_roles or (self.template and self.template.experiments[0].model_window is not None):
            return  # Timeline charging and support stages are carried into follow-up recipes.
        required = {role for role in self.roles if role not in {"T", "AS", "seed"}
                    and _role_active(campaign, self.roles[role])}
        if self.demo:
            required -= {"clean_sol", "valve_state"}
        traces = (campaign.doe_decisions or {}).get("traces", {}) if campaign else {}
        have_stirrer = self.mode == "bo" or "stirring" in traces
        missing = required - set(support) - ({"stirring_rate"} if have_stirrer else set())
        if missing:
            raise ValueError("Define follow-up support profiles for: " + ", ".join(sorted(missing)))
        if self.seed_rpm is not None and "seed" in support:
            raise ValueError("Assign a seed pump and choose either calibrated delivery or a support profile.")
        if not self.demo and "seed" in self.roles and _role_active(campaign, self.roles["seed"]):
            if self.seed_rpm is None and "seed" not in support:
                raise ValueError("Set calibrated seed delivery or a fixed seed support profile for follow-ups.")

    def _support_profile(self, role, settings):
        data = _merge({"points": None, "step": role == "valve_state"}, settings, f"{role} support profile")
        profile = make_profile(data["points"], "Configured support", step=data["step"])
        if profile is None or profile.time[0] != 0 or type(data["step"]) is not bool:
            raise ValueError(f"{role}: enter a timeline from zero as [times in s, values].")
        device = self.roles[role]
        config = self.configs[device]
        if config["type"] in {"masterflex", "ismatec", "nidaq_valve"} and profile.values[-1] != 0:
            raise ValueError(f"End the {role} support profile at zero.")
        profile = self.device_profile(role, profile)
        ExperimentDefinition.from_frame(ExperimentDefinition(role, (profile,)).to_frame(), self.configs)
        return profile

    def device_profile(self, role, profile, process=None):
        assignment = self.assignments[role]
        config = self.configs[assignment.device_id]
        times, values = profile.time, profile.values
        if role == "AS" and self.demo and config["type"] == "ismatec" and not config.get("pump_calibration"):
            values, unit = values * 60, "RPM"
        else:
            mass_flow = self.role_specs[role].unit == "g/h"
            if role == "AS":
                values = values * 3600
            if mass_flow:
                from experiment.doe_campaign import ProfileAssignment
                from experiment.pump_calibration_files import calibration_knots

                semantic = self.line_roles.get(role, {}).get("role", role)
                conversion = ProfileAssignment(semantic, assignment.device_id, assignment.density_g_ml)
                if not profile.step and config.get("pump_calibration"):
                    times, values = calibration_knots(config, times, values)
                values, unit, _ = convert_profile(self.calibration_storage, conversion, config, values)
            else:
                unit = target_limits(config)["unit"]
        return DeviceProfile(assignment.device_id, tuple(ProfilePoint(float(t), float(v))
                             for t, v in zip(times, values)),
                             "step" if profile.step or role == "valve_state" else "linear", unit)

    def build(self, recipe, process, *, name, source, default_seed, modeling=False):
        temp = recipe["temperature"]
        duration = float(temp.time[-1])
        if temp.time[0] != 0 or duration <= 0:
            raise ValueError("The follow-up temperature profile must start at zero and have a positive duration.")
        device_profiles = {"T": self.device_profile("T", temp)}
        feed = recipe.get("feed")
        if feed is not None:
            if feed.values.min() < 0 or feed.time[0] != 0 or feed.time[-1] != duration or feed.values[-1] != 0:
                raise ValueError("The antisolvent profile must be nonnegative, span the crystallization window and end at zero.")
            if np.any(feed.values > 0) and ("AS" not in self.roles or self.setup["recording"]["no_feed"]):
                raise ValueError("Provide an assigned and monitored antisolvent pump for the recommended recipe.")
            if "AS" in self.roles and not self.setup["recording"]["no_feed"]:
                device_profiles["AS"] = self.device_profile("AS", feed, process)
        stirring = recipe.get("stirring")
        support = {role: spec for role, spec in self.setup["support_profiles"].items()
                   if role != "stirring_rate" or stirring is None}
        if stirring is not None:
            if "stirring_rate" not in self.roles and np.any(stirring.values != 0):
                raise ValueError("Assign a main stirrer to execute the stirring recommendation.")
        if stirring is not None and "stirring_rate" in self.roles:
            device_profiles["stirring_rate"] = self.device_profile("stirring_rate", stirring)
        if ("stirring_rate" in self.roles and _role_active(self.template, self.roles["stirring_rate"])
                and "stirring_rate" not in device_profiles and "stirring_rate" not in support and not self.line_roles
                and not (self.template and self.template.experiments[0].model_window is not None)):
            raise ValueError("Provide a recorded or configured main stirrer profile for the follow-up.")
        seed_mass = _number(process.get("SeedMass", default_seed), "seed mass")
        if recipe.get("volume") is not None:
            requested = float(recipe["volume"].values[0])
            current = charge_volume(process)
            process.pop("VolumeFromDesignConditions", None)
            if current is not None and np.isclose(requested, current):
                process.pop("RequestedInitialVolume_ml", None)
            else:
                process["RequestedInitialVolume_ml"] = requested
                if not self.template or self.template.experiments[0].model_window is None:
                    raise ValueError("Volume decisions require defined preparation charging stages. Hold volume fixed or define and select the experiment period.")
        if not self.demo and (self.line_roles or (self.template and self.template.experiments[0].model_window is not None)):
            return self._timeline_followup(device_profiles, process, name, source, default_seed, duration)
        if self.demo:
            for role, profile in support.items():
                device_profiles[role] = self._support_profile(role, profile)
            demo = self._append_demo_profiles(device_profiles, duration, seed_mass, temp, feed)
        else:
            demo = None
            if modeling and seed_mass > 0 and not self.setup["recording"]["seed_percentiles"]:
                raise ValueError("Configure seed percentiles before executing a seeded follow-up.")
            if self.seed_rpm is not None:
                rpm, end = seed_dose_command(self.configs[self.roles["seed"]], self.setup["seed_pump"], seed_mass)
                if end >= duration:
                    raise ValueError("Seed addition exceeds the recommended experiment duration.")
                if seed_mass:
                    seed = Profile(np.array([0., end, duration]), np.array([rpm, 0., 0.]), "Calibrated seed dose", step=True)
                    device_profiles["seed"] = self.device_profile("seed", seed)
            elif seed_mass != default_seed:
                raise ValueError("Configure calibrated seed delivery before changing the preloaded seed mass.")
            start = self.setup["followup_start_s"]
            if start:
                for role, profile in device_profiles.items():
                    points = tuple(ProfilePoint(point.time_s + start, point.target) for point in profile.points)
                    before = [ProfilePoint(0., points[0].target if role in {"T", "stirring_rate"} else 0.)]
                    if role == "AS" and profile.interpolation == "linear" and points[0].target != 0:
                        transition = float(np.nextafter(start, 0.))
                        if transition > 0:
                            before.append(ProfilePoint(transition, 0.))
                    device_profiles[role] = DeviceProfile(profile.device_id, (*before, *points),
                                                          profile.interpolation, profile.unit)
            for role, profile in support.items():
                device_profiles[role] = self._support_profile(role, profile)
        profiles = tuple(device_profiles.values())
        experiment = ExperimentDefinition(name, profiles, source=source, imed_simulation=demo)
        ExperimentDefinition.from_frame(experiment.to_frame(), self.configs)
        start = 0. if self.demo else self.setup["followup_start_s"]
        return Campaign(experiment.name, (experiment,), monitor_device_ids=self.monitors), (start, start + duration)

    def _timeline_followup(self, modeled, process, name, source, default_seed, duration):
        from dataclasses import replace

        original = self.template.experiments[0]
        seed_mass = process.get("SeedMass", default_seed)
        seed_delivery = None
        original_seed = (original.model_process or {}).get("SeedMass", original.seed_mass_g)
        seed_device = self.roles.get("seed")
        seed_config = self.configs.get(seed_device, {})
        if original_seed is None and original.model_window and seed_config.get("seed_calibration"):
            seed_profile = next((p for p in original.profiles if p.device_id == seed_device
                                 and any(point.target > 0 for point in p.points)), None)
            if seed_profile is not None:
                from experiment.pump_calibration_files import seed_mass_from_profile

                original_seed = seed_mass_from_profile(seed_profile, seed_config, original.model_window[0])
        if not np.isclose(seed_mass, default_seed if original_seed is None else original_seed):
            if self.staged:
                raise ValueError("The recommendation changes the seed dose. Use the seed dose programmed in the starting recipe's preparation stages.")
            for template in self.template.experiments:
                stored = (template.model_process or {}).get("SeedMass", template.seed_mass_g)
                if template.model_window is not None and stored is not None and np.isclose(stored, seed_mass):
                    original = template
                    break
            else:
                if self.seed_rpm is None or original.model_window is None:
                    raise ValueError("Use a seed mass from a starting experiment, or configure calibrated seed delivery.")
                slot = next((stage for stage in original.run_stages if stage.purpose == "Seeding"), None)
                if slot is None or slot.end_s > original.model_window[0]:
                    raise ValueError("Define a Seeding stage before the modeling period to vary the seed dose.")
                seed_device = self.roles["seed"]
                rpm, seconds = seed_dose_command(self.configs[seed_device], self.setup["seed_pump"], seed_mass)
                stop = slot.start_s + seconds
                if stop > slot.end_s:
                    raise ValueError("The proposed seed mass does not fit the calibrated Seeding stage.")
                seed_delivery = (seed_device, slot.start_s, stop, rpm)
        if original.model_window is None:
            raise ValueError("Mark the Experiment and Equilibration stages in Experiment Definition before running a modeling loop.")
        begin, end = original.model_window
        new_end = begin + duration
        run_end = max(profile.points[-1].time_s for profile in original.profiles)
        from experiment.material_balance import initial_charges

        volumes = {"AS": "AntiSolvMass", "solvent": "SolvMass", "solution": "SolvMass"}
        charge_roles = {device: self.line_roles.get(role, {}).get("role", role) for role, device in self.roles.items()
                        if self.line_roles.get(role, {}).get("role", role) in volumes}
        scales = {}
        if "RequestedInitialVolume_ml" in process:
            reference_process = {**process, **(original.model_process or {})}
            reference_process.pop("InitialVolume_ml", None)
            inferred = initial_charges(original, reference_process, self.roles, self.configs, line_roles=self.line_roles)
            reference = inferred.get("InitialVolume_ml")
            if reference is None or reference <= 0:
                raise ValueError("Volume decisions require a known starting liquid volume.")
            if self.staged and not np.isclose(process["RequestedInitialVolume_ml"], reference):
                raise ValueError("The recommendation changes the starting liquid volume. Use the charges programmed in the starting recipe's preparation stages.")
            volume_scale = process["RequestedInitialVolume_ml"] / reference
            explicit_masses = process.get("VolumeFromDesignConditions")
            for key in ("SolvMass", "AntiSolvMass"):
                planned = process[key] if explicit_masses else inferred[key] * volume_scale
                if not np.isclose(planned, inferred[key]):
                    if self.staged:
                        raise ValueError("The recommendation changes the starting liquid charges. Use the charges programmed in the preparation stages.")
                    prepared = inferred.get("PreparationCharges_g", {})
                    if inferred[key] <= 0 or not any(prepared.get(role, 0) > 0 for role in volumes if volumes[role] == key):
                        raise ValueError(f"Varying {PROCESS_FIELDS[key].lower()} needs a defined preparation charge for that liquid.")
                    scales[key] = planned / inferred[key]
                process[key] = planned
            process["InitialVolume_ml"] = process["RequestedInitialVolume_ml"]
        shift = begin + duration - end
        by_device = {profile.device_id: profile for profile in modeled.values()}
        profiles = []
        for profile in original.profiles:
            if seed_delivery and profile.device_id == seed_delivery[0]:
                _, start, stop, rpm = seed_delivery
                seed_points = {0.: 0., begin: 0.}
                if stop > start:
                    seed_points.update({start: rpm, stop: 0.})
                seed_points.update({p.time_s: p.target for p in profile.points if p.time_s > begin})
                profile = DeviceProfile(profile.device_id, tuple(ProfilePoint(t, v) for t, v in sorted(seed_points.items())),
                                        "step", profile.unit)
            if (mass_key := volumes.get(charge_roles.get(profile.device_id))) in scales:
                from experiment.pump_calibration_files import delivered_flow, convert_flow
                cfg = self.configs[profile.device_id]
                scale = scales[mass_key]
                def scaled(target):
                    if cfg.get("pump_calibration"):
                        return float(convert_flow(cfg["pump_calibration"], cfg["type"], delivered_flow(cfg, [target]) * scale)[0])
                    if cfg["type"] == "ismatec" and cfg.get("simulated"):
                        return target * scale
                    raise ValueError(f"Load the charging calibration for {profile.device_id} before varying initial volume.")
                profile = DeviceProfile(profile.device_id,
                    tuple(ProfilePoint(p.time_s, scaled(p.target) if p.time_s < begin else p.target) for p in profile.points),
                    profile.interpolation, profile.unit)
            replacement = by_device.get(profile.device_id)
            if replacement is not None:
                profiles.append(_replace_model_window(profile, replacement, begin, end, new_end, run_end=run_end,
                                                      suffix_starts_at_boundary=self.staged))
                continue
            if self.staged:
                trace = Profile(np.asarray([p.time_s for p in profile.points]),
                                np.asarray([p.target for p in profile.points]), "Starting recipe",
                                step=profile.interpolation == "step")
                times = np.union1d(trace.time, [begin, end])
                points = tuple(ProfilePoint(_followup_time(t, begin, end, new_end), float(value))
                               for t, value in zip(times, trace.sample(times)))
                profiles.append(replace(profile, points=points))
                continue
            if any(begin + duration < point.time_s < end for point in profile.points):
                raise ValueError(f"The recommended duration is shorter than {profile.device_id}'s support sequence.")
            points = tuple(ProfilePoint(p.time_s + shift if p.time_s >= end else p.time_s, p.target)
                           for p in profile.points)
            profiles.append(replace(profile, points=points))
        if seed_delivery and all(profile.device_id != seed_delivery[0] for profile in original.profiles):
            device, start, stop, rpm = seed_delivery
            if stop > start:
                points = {0.: 0., begin: 0., start: rpm, stop: 0.}
                profiles.append(DeviceProfile(device, tuple(ProfilePoint(t, v) for t, v in sorted(points.items())), "step", "RPM"))
        modes = tuple(replace(mode, time_s=_followup_time(mode.time_s, begin, end, new_end))
                      for mode in original.control_modes)
        from experiment.run_stages import RunStage
        stages = tuple(replace(stage, start_s=_followup_time(stage.start_s, begin, end, new_end),
                               end_s=_followup_time(stage.end_s, begin, end, new_end))
                       for stage in original.run_stages)
        if not stages:
            stages = (RunStage("modeled-experiment", f"{self.mode.upper()} proposed experiment", "Experiment", begin, new_end),)
        experiment = ExperimentDefinition(name, tuple(profiles), source=source, control_modes=modes,
                                          model_window=(begin, new_end), run_stages=stages,
                                          initial_mass_g=original.initial_mass_g, seed_mass_g=seed_mass)
        if self.setup["derive_initial_charges"] or scales:
            actual = initial_charges(experiment, process, self.roles, self.configs, line_roles=self.line_roles)
            visible = actual.get("PreparationCharges_g", {})
            if visible:
                for key in ("SolvMass", "AntiSolvMass"):
                    if key in scales and not np.isclose(actual[key], process[key], rtol=.002, atol=1e-6):
                        raise ValueError(f"Converted pump profiles deliver {actual[key]:g} g for {PROCESS_FIELDS[key].lower()}. "
                                         f"The recommendation requested {process[key]:g} g.")
                process.update(actual)
        if self.simulation and not self.demo:
            simulation = calibrated_simulation(self.simulation, self.roles, self.configs, self.monitors, experiment)
            model_process = {**process, "SeedPercentiles": list(self.setup["recording"]["seed_percentiles"])}
            experiment = replace(experiment, imed_simulation=simulation, model_process=model_process)
        ExperimentDefinition.from_frame(experiment.to_frame(), self.configs)
        return Campaign(name, (experiment,), monitor_device_ids=self.monitors, line_roles=self.line_roles), experiment.model_window

    def _append_demo_profiles(self, profiles, duration, seed_mass, temp, feed):
        _, _, seeded_process, _, _ = model_inputs(True)
        if seed_mass and not np.isclose(seed_mass, seeded_process["SeedMass"]):
            raise ValueError("The validation simulator supports its calibrated seed dose or zero seed.")
        demo_template = next((exp for exp in self.template.experiments
                              if exp.imed_simulation["seeded"] == bool(seed_mass)), None)
        if demo_template is None:
            raise ValueError("The simulated initial batch needs a matching seed-dose profile.")
        original = demo_template.imed_simulation
        last_change = max(temp.time[np.flatnonzero(np.diff(temp.values) != 0) + 1], default=0.)
        final_feed = 0.
        if feed is not None and np.any(feed.values > 0):
            last = int(np.flatnonzero(feed.values > 0)[-1])
            final_feed = feed.time[min(last + 1, len(feed.time) - 1)]
        hold = float(max(last_change, final_feed, duration / 2))
        if hold >= duration:
            raise ValueError("The simulated recommendation needs a final hold at constant temperature and zero feed.")
        for role, trace in (("T", temp), ("AS", feed)):
            if trace is None:
                trace = Profile(np.array([0., duration]), np.zeros(2), "No antisolvent")
            time = np.union1d(trace.time, [hold, duration])
            if trace.step:
                changes = np.flatnonzero(np.diff(trace.values) != 0) + 1
                before = np.nextafter(trace.time[changes], trace.time[changes - 1])
                time = np.union1d(time, before)
            profiles[role] = self.device_profile(role, Profile(time, trace.sample(time), trace.source, trace.step))
        for role in ("seed", "clean_sol", "valve_state"):
            if role in profiles:
                continue
            source = next(p for p in demo_template.profiles if p.device_id == self.roles[role])
            if role == "seed":
                points = tuple(p for p in source.points if p.time_s < hold)
                if not points or points[-1].target != 0:
                    raise ValueError("Seed delivery extends into simulated equilibration.")
            else:
                points = (ProfilePoint(0., 0.), ProfilePoint(duration, 0.), *(
                    ProfilePoint(duration + p.time_s - original["end_s"], p.target)
                    for p in source.points if p.time_s > original["end_s"] + 1e-8))
            profiles[role] = DeviceProfile(source.device_id, points, source.interpolation, source.unit)
        spec = {**original, "end_s": duration, "hold_start_s": hold, "label": f"{self.mode.upper()} recommendation"}
        return compile_simulation(spec, self.roles, self.configs, self.monitors, None, tuple(profiles.values()))


class AdaptiveLoop:
    def __init__(self, mode, setup, campaign, configs, connected, calibration_storage, assignments, case,
                 *, bo_settings=None, size_term, workers=1, final_size_samples=1):
        self.mode = mode
        self.setup = validate_setup(setup)
        self.compiler = RecipeCompiler(self.setup, campaign, configs, calibration_storage, assignments,
                                       campaign.monitor_device_ids, mode)
        campaign = self.compiler.template
        self.roles = self.compiler.roles
        self.configs = self.compiler.configs
        self.demo = self.compiler.demo
        self.template = campaign
        self.case = deepcopy(case)
        self.bo = deepcopy(bo_settings)
        if self.bo is not None:
            self.bo["discrete_seed"] = bool(self.demo or self.compiler.seed_rpm is None)
        self.size_term, self.workers = size_term, workers
        self.final_size_samples = validate_final_size_samples(final_size_samples)
        self.followups = 0
        self.process_by_id = {}
        self.windows = {}
        self.used_recordings = set()
        self.doe_template_index = None
        self.recipe_origins = {}
        unmapped = {profile.device_id for experiment in campaign.experiments for profile in experiment.profiles}
        unmapped -= set(self.roles.values())
        if unmapped:
            raise ValueError("Assign the initial actuators to known profile roles: " + ", ".join(sorted(unmapped)))
        self.check_devices(campaign, configs, connected)
        self._prepare_measurements(campaign)
        self.doe_decisions = planned_decisions(self.template, self.setup, self.roles, self.configs, self.case)
        self.template = replace(self.template, doe_decisions=self.doe_decisions)
        self.compiler.template = self.template
        expected_names = {experiment.name for experiment in campaign.experiments}
        unknown = self.setup["initial_processes"].keys() - expected_names
        if unknown:
            raise ValueError("Unknown initial experiment names: " + ", ".join(sorted(unknown)))
        if self.mode == "bo":
            self._prepare_bo(campaign)
        else:
            self._prepare_imed()
            self.active_models = tuple(range(len(self.case["ModelParameter"])))
            self.stage_counts = Counter(stage["kind"] for stage in loop_stages(self.case))
        self.compiler.check_support(self.template)

    @property
    def complete(self):
        return self.followups >= self.setup["max_followups"]

    def check_devices(self, campaign, configs, connected):
        self.compiler.check_devices(campaign, configs, connected, require_all=True)

    def _prepare_measurements(self, campaign):
        if self.demo:
            _, _, self.process, solubility, mesh = model_inputs(self.demo["seeded"])
            self.case.setdefault("Solubility", solubility)
            self.case["pbmsetting"] = {**mesh, **self.case.get("pbmsetting", {})}
            self.mapping = {}
            return
        defaults = self.compiler.process_defaults(self.case)
        if self.setup["recording"]["initial_concentration"] is not None:
            defaults.setdefault("InitialConcentration", self.setup["recording"]["initial_concentration"])
        prepared = []
        for experiment in campaign.experiments:
            try:
                process = _process(starting_run_process(experiment, self.setup, self.roles, self.configs,
                                                       self.compiler.line_roles, defaults))
            except ValueError as error:
                raise ValueError(f"{experiment.name}: {error}") from error
            if "InitialConcentration" in process:
                _number(process["InitialConcentration"], "initial concentration", 0, 1 - 1e-12)
            elif experiment.imed_simulation:
                raise ValueError(f"{experiment.name}: enter the initial concentration for the staged PBM simulation.")
            sizes = self.setup["recording"]["seed_percentiles"]
            if process["SeedMass"] and (not sizes or not np.all(np.asarray(sizes) > 0)):
                raise ValueError("Positive seed mass requires positive seed D10/D50/D90. Use 0 g seed and 0/0/0 for an unseeded run.")
            if self.mode == "imed":
                process["MixingSettings"] = self.setup["mixing"]
                if not self.setup["mixing"]["enabled"]:
                    process["PowerRatio"] = 1.
            self.process_by_id[experiment.experiment_id] = process
            initial = process["SolvMass"] + process["AntiSolvMass"]
            concentration = process.get("InitialConcentration", 0.) or 0.
            initial /= 1 - concentration
            if process.get("PreparationCharges_g"):
                initial = 0.
            elif "seed" not in self.roles:
                initial += process["SeedMass"]
            modeled_fields = (*POSITIVE_PROCESS, *NONNEGATIVE_PROCESS, "InitialConcentration", "InitialVolume_ml", "MixingSettings")
            prepared.append(replace(experiment, initial_mass_g=initial,
                                    model_process={**{key: process[key] for key in modeled_fields if key in process},
                                                   "SeedPercentiles": list(sizes)}))
            if experiment.model_window is not None:
                self.windows[experiment.experiment_id] = experiment.model_window
        self.process = dict(self.process_by_id[campaign.experiments[0].experiment_id])
        self.template = replace(campaign, experiments=tuple(prepared))
        self.compiler.template = self.template
        required = imed_objective_channels(imed_response_weights(self.case)) if self.mode == "imed" else None
        self.mapping = recording_mapping(self.setup, self.roles, self.configs, required_measurements=required)
        self._require_channels("Temperature")
        recording = self.setup["recording"]
        if any("InitialConcentration" not in process for process in self.process_by_id.values()) and not self.mapping.get("Concentration"):
            raise ValueError("Enter Initial concentration in Recording or Process inputs, or map a concentration channel sampled at the start.")
        source = antisolvent_recording_source(recording, self.mapping, campaign, self.roles, self.configs,
                                             infer_no_feed=self.mode == "imed")
        if source == "No antisolvent feed":
            recording["no_feed"] = True
        if self.mode == "imed" and self.setup["mixing"]["enabled"] and "stirring_rate" not in self.roles:
            raise ValueError("Assign the main stirrer through MSI or Devices to calculate average mixing power.")

    def _require_channels(self, *roles):
        missing = [role for role in roles if not self.demo and not self.mapping.get(role)]
        if missing:
            raise ValueError("Map the required recorded channels: " + ", ".join(missing))

    def _prepare_bo(self, campaign):
        if find_spec("sklearn") is None:
            raise ValueError("Install the BO runtime from requirements-bo.txt or use a Windows build with BO.")
        if len(self.bo["selected"]) + len(campaign.experiments) < 2:
            raise ValueError("BO needs at least two observations across the initial batch and selected prior runs.")
        objectives = {key for key, target in self.bo["targets"].items() if target["enabled"] and target["weight"] > 0}
        outcomes = objectives | {key for key, bounds in self.bo["output_bounds"].items() if any(v is not None for v in bounds)}
        if not objectives:
            raise ValueError("Enable a BO target with positive weight.")
        if outcomes & {"d10", "d50", "d90"}:
            self._require_channels("D10", "D50", "D90")
        if "yield" in outcomes:
            if not self.case.get("Solubility") and self.bo["yield_basis"] == "theoretical":
                raise ValueError("Load the applicable IMED solubility for theoretical yield, or select actual yield.")
            if self.bo["yield_basis"] == "actual":
                self._require_channels("Concentration")

    def _prepare_imed(self):
        if not self.case.get("ModelParameter") and self.demo:
            self.case["ModelParameter"] = [validation_model()]
            self.case.setdefault(PYFIT_SETTINGS_KEY, dict(VALIDATION_PE_SETTINGS))
        coefficients = self.case.get("Solubility")
        if not isinstance(coefficients, dict):
            raise ValueError("Open Solubility and enter coefficients a–h before starting IMED.")
        for key in "abcdefgh":
            _number(coefficients.get(key), f"solubility coefficient {key} under Solubility", -np.inf)
        if not self.case.get("ModelParameter"):
            raise ValueError("Add a candidate kinetic model under IMED settings before starting IMED.")
        for index, model in enumerate(self.case["ModelParameter"], 1):
            table = np.asarray(model.get("paraset", []), dtype=float)
            if table.shape != (35, 4):
                raise ValueError(f"Set valid parameter bounds and enable estimation in IMED model {index}.")
            active = table[table[:, 0] == 1, 1:]
            if (not len(active) or not np.isfinite(active).all() or np.isnan(table).any()
                    or np.isposinf(table).any() or np.any(active[:, 1] > active[:, 2])):
                raise ValueError(f"Set valid parameter bounds and enable estimation in IMED model {index}.")
        self.weights = imed_response_weights(self.case)
        if not any(self.weights):
            raise ValueError("Open IMED objectives and set at least one positive weight for FBRM, concentration or D-values.")
        self._require_channels(*imed_objective_channels(self.weights))
        self.pe_mesh = effective_pe_pbm(self.case)
        pbm_backend_label(self.pe_mesh)
        prior = self.case.get("ExpData", [])
        if prior:
            FitObjective(prior, self.case.get("ProcInf", []), self.case["Solubility"],
                         self.pe_mesh, self.weights, self.size_term)
        else:
            # A process-only template supplies defaults without becoming a fit observation.
            self.case["ProcInf"] = []
        options = self.setup["imed"]
        design_settings = self.case.get("pbmsetting", {})
        self.case["pbmsetting"] = {**self.pe_mesh, **(design_settings if isinstance(design_settings, dict) else {})}
        if options["use_doe_decisions"] and self.doe_decisions and self.doe_decisions["traces"].get("temperature"):
            if self.demo and options["equilibration_s"] <= 0:
                raise ValueError("Add a positive equilibration duration for simulated IMED follow-ups.")
            return
        conditions = self._imed_followup_conditions(self.process)
        if not conditions:
            raise ValueError("Enter automatic IMED design conditions in Setup.")
        if self.demo and options["equilibration_s"] <= 0:
            raise ValueError("Add a positive equilibration duration for simulated IMED follow-ups.")
        addition, bounds = recipe_space(conditions)
        temperature_limits = target_limits(self.configs[self.roles["T"]])
        if not all(target_allowed(temp, temperature_limits) for temp in conditions["temperature"]):
            raise ValueError("IMED design temperatures exceed the assigned temperature controller limits.")
        if addition and "AS" not in self.roles:
            raise ValueError("Assign an antisolvent pump for the IMED design.")
        if addition:
            rates = np.array([addition / duration for duration in bounds[3]])
            feed = self.compiler.device_profile("AS", Profile(np.array([0., 1.]), rates, "Design bounds"), self.process)
            feed_limits = target_limits(self.configs[feed.device_id])
            if not all(target_allowed(point.target, feed_limits) for point in feed.points):
                raise ValueError("IMED antisolvent rate bounds exceed the assigned pump limits.")
        if conditions["seed_bounds"][1] > 0 and not self.demo:
            if not self.setup["recording"]["seed_percentiles"]:
                raise ValueError("Enter seed percentiles for a seeded IMED design.")
            if self.compiler.seed_rpm is None and list(conditions["seed_bounds"]) != [self.process["SeedMass"]] * 2:
                raise ValueError("Configure calibrated seed delivery or fix IMED seed mass to the preloaded mass.")
        if self.demo:
            _, simulated, seeded, _, _ = model_inputs(True)
            concentration = simulated["ConcentrationData"][1][0]
            if (not np.allclose(conditions["liquid_masses"], [self.process["SolvMass"], self.process["AntiSolvMass"]])
                    or not np.isclose(conditions["concentration"], concentration)):
                raise ValueError("Use the validation simulator's starting liquid masses and concentration in IMED design.")
            low, high = conditions["seed_bounds"]
            available = [seeded["SeedMass"] if item.imed_simulation["seeded"] else 0.
                         for item in self.template.experiments]
            if low != high or not any(np.isclose(low, mass) for mass in available):
                raise ValueError("Fix simulated IMED seed mass to a dose available in the validation DoE.")

    def _imed_followup_conditions(self, process, experiment=None):
        options = self.setup["imed"]
        conditions = dict(options["conditions"])
        if conditions and self.compiler.staged and not self.demo:
            concentration = (experiment or {}).get("InitialConcentration", process.get("InitialConcentration"))
            conditions.update(liquid_masses=(process["SolvMass"], process["AntiSolvMass"]),
                              seed_bounds=(process["SeedMass"], process["SeedMass"]))
            if concentration is not None:
                conditions["concentration"] = concentration
        return imed_design_conditions(conditions, no_feed=self.setup["recording"]["no_feed"])

    def _read_batch(self, manifest, target, report):
        for experiment_id in manifest["experiments"]:
            if window := self.windows.get(experiment_id):
                report(f"Importing only the modeling experiment period {window[0]:g}–{window[1]:g} s; "
                       "full preparation and cleanup recordings are retained.")
            process = self.process_by_id.get(experiment_id, {})
            if charges := process.get("PreparationCharges_g"):
                summary = ", ".join(f"{role} {mass:g} g" for role, mass in charges.items())
                report(f"Calibrated preparation charges: {summary}. Initial solvent {process['SolvMass']:g} g, "
                       f"antisolvent {process['AntiSolvMass']:g} g, dry seed {process['SeedMass']:g} g.")
        experiments, processes, runs = read_completed_recordings(
            Path(manifest["campaign_folder"]), manifest["experiments"], self.setup["recording"],
            self.mapping, self.process_by_id, self.windows, roles=self.roles,
            mixing=self.setup["mixing"] if self.mode == "imed" and not self.demo else None,
            allow_density_fallback=self.mode == "imed", final_size_samples=self.final_size_samples)
        if self.final_size_samples > 1:
            for experiment in experiments:
                report(f"{experiment['Name']}: used {experiment['FinalSizeSamplesUsed']} of "
                       f"{self.final_size_samples} requested D10 / D50 / D90 samples for final sizes.")
        if self.mode == "imed":
            for experiment, process in zip(experiments, processes):
                if not self.demo:
                    report(f"{experiment['Name']}: additions from {experiment.get('AntisolventSource')}; "
                           f"PowerRatio={process['PowerRatio']:.8g}; average mixing inputs {process.get('MixingSummary', {})}.")
        loop_outputs.saved_recordings(target, runs, report)
        return experiments, processes

    def recommend(self, manifest, cancelled, report, *, publish=None, collect_only=False):
        publish = publish or (lambda stage, payload: None)

        def progress(message):
            if cancelled():
                raise InterruptedError("Modeling stopped.")
            report(message)

        if manifest["campaign_id"] in self.used_recordings:
            raise ValueError("This campaign was already modeled.")
        target = Path(manifest["model_path"])
        progress(f"Importing {len(manifest['experiments'])} completed recordings into {self.mode.upper()}")
        experiments, processes = self._read_batch(manifest, target, progress)
        if not self.used_recordings:
            from experiment.campaign_effects import initial_input_groups, power_effects
            groups = initial_input_groups(processes)
            # Older scale-only pump setups may have no command-to-mass curve.
            # The recording can supply that missing profile, with the planned
            # decisions taking precedence for channels that have command data.
            measured = capture_decisions(case_runs({"ExpData": experiments, "ProcInf": processes},
                                                   manifest["campaign_folder"], controls=self.roles, prefer_commands=True))
            planned = self.doe_decisions or {"traces": {}, "levels": {}}
            self.doe_decisions = {key: {**measured[key], **planned[key]} for key in planned}
            workflow = self.case.setdefault("PythonWorkflow", {})
            workflow.update(initial_input_groups=groups, doe_decisions=self.doe_decisions)
            if self.bo is not None:
                self.bo["initial_input_groups"] = groups
                self.bo["doe_decisions"] = self.doe_decisions
            if not self.demo:
                self.process.update(processes[0])
            progress(f"Initial DoE decision variables: {decision_summary(self.doe_decisions)}.")
            if self.mode == "imed" and self.setup["mixing"]["enabled"]:
                reports = power_effects(self.case["ModelParameter"], processes)
                self.case["PythonWorkflow"]["mixing_effects"] = reports
                for report in reports:
                    progress(f"IMED model {report['model']}: estimating power effects {report['estimated_power_effects']}. {report['note']}")
        if self.mode == "bo":
            state = self._collect_bo(experiments, processes, manifest)
            loop_outputs.bo_inputs(target, state)
            publish("bo_inputs", state)
        else:
            origin = self.recipe_origins.get(manifest["campaign_id"], {})
            for experiment in experiments:
                experiment.update(origin)
            self.case["ExpData"] = [*self.case.get("ExpData", []), *experiments]
            self.case["ProcInf"] = [*self.case.get("ProcInf", []), *processes]
            self.case[PE_WEIGHTS_KEY] = {**{f"W{i}": weight for i, weight in enumerate(self.weights, 1)},
                                         "Terms": self.size_term}
            loop_outputs.imed_inputs(target, self.case)
            publish("imed_inputs", {"case": deepcopy(self.case), "folder": manifest["campaign_folder"]})
        progress(f"Saved {self.mode.upper()} inputs and recording CSVs to {target}")
        if self.mode == "imed":
            survivors, scores, pe_stage = self._fit_imed(cancelled, report, publish, target)
        if collect_only:
            if self.mode == "imed":
                progress("Follow-up limit reached. The final IMED fit includes every completed experiment.")
            else:
                self._save_bo_stage(target, progress, publish)
                progress("Follow-up limit reached. Final measurements were added to the modeling inputs.")
            self.used_recordings.add(manifest["campaign_id"])
            return None
        if self.mode == "bo":
            result, process, details = self._recommend_bo(state, progress, publish)
            profiles = result.profiles
            stage = self._save_bo_stage(target, progress, publish, proposal=result)
            details.update(stage_name=stage["name"], stage_case=stage["path"])
        else:
            if not survivors:
                raise ValueError("No IMED candidate passed model screening. Review the saved PE results and model diagnostics.")
            profiles, process, details = self._recommend_imed(
                experiments, processes, survivors, scores, pe_stage, cancelled, progress, publish, target)
        progress("Validating the recommended experiment")
        if self.compiler.staged and not self.demo:
            progress("Applying the recommendation only inside the selected experiment period. Reusing the programmed preparation and following stages.")
        _process(process)
        campaign, window = self.compiler.build(
            profiles, process, name=f"{self.mode.upper()} follow-up {self.followups + 1}",
            source=self.mode, default_seed=self.process["SeedMass"], modeling=True)
        if (stirrer := profiles.get("stirring")) is not None:
            process["StirringRPM"] = stirrer.total() / (stirrer.time[-1] - stirrer.time[0])
        process_fields = (*POSITIVE_PROCESS, *NONNEGATIVE_PROCESS, "InitialConcentration", "InitialVolume_ml", "StirringRPM", "MixingSettings")
        experiment = replace(campaign.experiments[0], seed_mass_g=float(process["SeedMass"]), model_window=window,
                             model_process={**{key: process[key] for key in process_fields if key in process},
                                            "SeedPercentiles": list(self.setup["recording"]["seed_percentiles"])})
        campaign = replace(campaign, experiments=(experiment,))
        self.process_by_id[experiment.experiment_id] = process
        self.windows[experiment.experiment_id] = window
        from experiment.pump_corrections import corrected_campaign

        executed = corrected_campaign(campaign, self.configs, self.windows)
        process_state = {key: float(process[key]) for key in (*POSITIVE_PROCESS, *NONNEGATIVE_PROCESS)}
        for key in ("InitialConcentration", "InitialVolume_ml", "StirringRPM"):
            if process.get(key) is not None:
                process_state[key] = float(process[key])
        payload = {"format": "crystaloop-recommendation", "version": 1, "backend": self.mode,
                   "input_campaign_id": manifest["campaign_id"], "campaign": asdict(executed),
                   "process": process_state, "crystallization_window_s": executed.experiments[0].model_window or window,
                   **details}
        if self.mode == "imed":
            save_native_case(target / "fit.imed.json", self.case)
            self.active_models = survivors
        self.recipe_origins[campaign.campaign_id] = {
            "RecipeSource": details["method"], "RecipeReason": details.get("design_reason", ""),
            "RecipeStage": details["stage_name"], "RecipeModelFolder": str(target),
        }
        loop_outputs.modeled_recipe(target, profiles, executed, details.get("predictions"))
        atomic_json(target / "recommendation.json", payload)
        progress(f"Saved the {details['method']} recommendation and device-target CSVs to {target}")
        self.used_recordings.add(manifest["campaign_id"])
        return campaign

    def _collect_bo(self, experiments, processes, manifest):
        from bo.physical_fit import initial_spec

        case = {"ExpData": experiments, "ProcInf": processes, "Solubility": self.case.get("Solubility")}
        added = case_runs(case, manifest["campaign_folder"], controls=self.roles)
        if origin := self.recipe_origins.get(manifest["campaign_id"]):
            for run in added:
                run["recipe_stage"] = origin["RecipeStage"]
        indices = range(len(self.bo["runs"]), len(self.bo["runs"]) + len(added))
        self.bo["runs"].extend(added)
        self.bo["selected"].extend(indices)
        selected = [self.bo["runs"][index] for index in self.bo["selected"]]
        for channel in ("temperature", "feed", "stirring"):
            if channel not in self.bo["physical_profiles"]:
                spec = initial_spec(channel, selected)
                if spec is not None:
                    self.bo["physical_profiles"][channel] = spec
        self.bo["source"] = manifest["campaign_folder"]
        return deepcopy(self.bo)

    def _recommend_bo(self, state, report, publish):
        from bo.optimizer import propose

        report("BO is fitting response models and searching for the next experiment")
        fixed = {}
        if self.compiler.staged and not self.demo:
            fixed["seed"] = self.process["SeedMass"]
            if (volume := charge_volume(self.process)) is not None:
                fixed["volume"] = volume
        result = propose(state, report=report, fixed_inputs=fixed)
        publish("bo_result", result)
        seed = result.profiles.get("seed")
        process = {**self.process, "SeedMass": float(seed.values[0]) if seed is not None else self.process["SeedMass"]}
        return result, process, {"method": "BO", "acquisition": result.acquisition,
                                 "predictions": result.predictions, "observations": len(state["selected"])}

    def _save_bo_stage(self, target, report, publish, *, proposal=None):
        entries = self.bo.setdefault("loop_stages", [])
        number = len(entries) + 1
        path = target / f"bo_{number:02}.json"
        entry = {"name": f"BO {number}", "path": str(path), "observations": len(self.bo["selected"]),
                 "proposed": proposal is not None}
        snapshot = {**self.bo, "loop_stages": [*entries, entry]}
        if proposal is not None:
            snapshot["loop_proposal"] = {
                **recipe_data(proposal.profiles, proposal.stages, proposal.horizon),
                "predictions": proposal.predictions, "acquisition": proposal.acquisition,
                "weighted_loss": proposal.mean_loss, "weighted_loss_sd": proposal.loss_sd,
            }
        atomic_json(path, snapshot)
        entries.append(entry)
        report(f"Saved {entry['name']} with {entry['observations']} observations to {path}.")
        publish("bo_stage", entry)
        return entry

    def _imed_stage(self, kind, target, candidates):
        self.stage_counts[kind] += 1
        number = self.stage_counts[kind]
        return {"kind": kind, "name": f"{kind} {number}", "experiments": len(self.case["ExpData"]),
                "case_file": str(target / f"{kind.lower()}_{number:02}.imed.json"), "candidates": candidates}

    def _fit_imed(self, cancelled, report, publish, target):
        publish("imed_pe", deepcopy(self.case))
        solubility = self.case["Solubility"]
        objective = FitObjective(self.case["ExpData"], self.case["ProcInf"], solubility,
                                 self.pe_mesh, self.weights, self.size_term, cancelled=cancelled)
        optimizer = self.case.get(PYFIT_SETTINGS_KEY, {})
        options = {key: optimizer.get(key, default) for key, default in PYFIT_DEFAULTS.items()}
        fits, models = {}, list(self.case["ModelParameter"])
        for position, index in enumerate(self.active_models, 1):
            if cancelled():
                break
            model = models[index]
            name = display_name(model, f"Model {index + 1}")
            report(f"IMED parameter estimation · {name} · {position} of {len(self.active_models)} active candidates")
            fit = fit_model(model, objective, workers=self.workers, cancelled=cancelled, progress=report,
                            retain_partial=True,
                            generation_progress=lambda generation, mean, best, i=index:
                            publish("imed_generation", (i, generation, mean, best)), **options)
            fits[index] = fit
        if not fits:
            raise InterruptedError("Stopped before a usable parameter evaluation finished.")
        assessment = screen_models(fits, sample_count=objective.sample_count, delta_aic=self.setup["imed"]["delta_aic"])
        survivors = tuple(index for index, result in assessment.items() if not result["reasons"])
        for index, fit in fits.items():
            result = {**pack_fit(fit), "screening": pack_screening(assessment[index])}
            record_step(self.case, "parameter_estimation", result, model_index=index,
                        context=pe_signatures(self.case, index)[0])
        updated = list(models)
        for index, fit in fits.items():
            updated[index] = {**models[index], "para": fit.parameters, "covmat": fit.covariance,
                              "SSR": fit.ssr, "AICVector": fit.aic, **matlab_screening_fields(assessment[index])}
        self.case["ModelParameter"] = updated
        candidates = [{"model": index + 1, "name": display_name(models[index], f"Model {index + 1}"),
                       "retained": index in survivors, "reason": ", ".join(result["reasons"]),
                       **pack_screening(result)} for index, result in assessment.items()]
        stage = self._imed_stage("PE", target, candidates)
        record_step(self.case, "model_screening", {"stage": stage})
        stage_path = save_native_case(stage["case_file"], self.case)
        report(f"Saved {stage['name']} with {len(survivors)} retained candidate(s) to {stage_path}.")
        destination = save_native_case(target / "fit.imed.json", self.case)
        report(f"Saved fitted IMED models to {destination}.")
        publish("imed_fits", {"fits": fits, "case": deepcopy(self.case), "screening": assessment})
        loop_outputs.imed_fits(target, models, fits, assessment)
        report(f"Saved IMED parameter and covariance CSVs to {target / 'imed'}.")
        for index, result in assessment.items():
            decision = ", ".join(result["reasons"]) or "retained"
            difference = result["delta_aic"]
            aic = f"ΔAIC {difference:.3g}" if difference is not None else "ΔAIC unavailable"
            report(f"IMED model {index + 1} · {aic} · F {result['f_value']:.3g}/{result['f_limit']:.3g} · {decision}")
        scores = {index: result["total_aic"] for index, result in assessment.items() if result["total_aic"] is not None}
        if cancelled():
            raise InterruptedError(f"Stopped. Current PE results saved to {destination}.")
        return survivors, scores, stage

    def _recommend_imed(self, experiments, processes, survivors, scores, pe_stage, cancelled, report, publish, target):
        from imed_python.doe_design import design_note, doe_controls, imed_doe_groups, imed_recipe_profiles, seed_template

        fitted = self.case["ModelParameter"]
        solubility = self.case["Solubility"]
        cutoff = self.setup["imed"]["delta_aic"]
        settings = self.case["pbmsetting"]
        settings = {**settings, "N": int(settings.get("N", 250))}
        if len(survivors) > 1:
            method = "Model discrimination"
            kind = "MD"
            reason = f"{len(survivors)} candidates passed the F-test, parameter t-tests and ΔAIC cutoff ({cutoff:g})."
            score = partial(discrimination_score, [fitted[index] for index in survivors], solubility=solubility,
                            settings=settings, weights=self.weights, size_term="D10 D50 D90",
                            experiment_count=len(self.case["ExpData"]))
        else:
            method = "D-optimal design"
            kind = "DO"
            index = survivors[0]
            model = fitted[index]
            name = display_name(model, f"Model {index + 1}")
            reason = f"Only {name} passed model screening."
            report(f"IMED selected {method} and skipped model discrimination. {reason}")
            samples = parameter_samples(model, self.setup["imed"]["covariance_samples"])
            size_term = self.size_term if sum(weight > 0 for weight in self.weights) == 1 else "CSD"
            score = partial(d_optimal_log_score, model, solubility=solubility, settings=settings,
                            enabled_terms=[weight > 0 for weight in self.weights], samples=samples, size_term=size_term)
        if len(survivors) > 1:
            report(f"IMED selected {method}. {reason}")
        options = self.setup["imed"]
        design_experiment = {**experiments[-1]}
        design_process = processes[-1]
        use_doe = options["use_doe_decisions"] and self.doe_decisions is not None
        staged = self.compiler.staged and not self.demo
        if staged or use_doe:
            if self.doe_template_index is None:
                self.doe_template_index = len(self.case["ExpData"]) - len(experiments)
            start = self.doe_template_index
            design_experiment = self.case["ExpData"][start]
            design_process = self.case["ProcInf"][start]
        conditions = {**self._imed_followup_conditions(design_process, design_experiment),
                      "equilibration_s": options["equilibration_s"]}
        if use_doe:
            groups = imed_doe_groups(self.doe_decisions, [fitted[index] for index in survivors],
                                     design_experiment, design_process)
            if staged:
                groups -= {"seed", "volume"}
            controls = doe_controls(self.doe_decisions, groups, discrete_seed=bool(self.demo or self.compiler.seed_rpm is None))
            conditions["doe_search"] = {"space": self.doe_decisions, "controls": controls,
                                        "seed_template": seed_template(self.case),
                                        "final_hold_s": options["equilibration_s"] if self.demo else 0.}
            report(design_note(self.doe_decisions, controls))
        elif conditions["seed_bounds"][1] > 0:
            design_experiment["InitialCSDsData"] = (model_inputs(True)[1]["InitialCSDsData"] if self.demo
                                                    else seed_distribution(self.setup["recording"]["seed_percentiles"]))
        template_index = self.doe_template_index if self.doe_template_index is not None else len(self.case["ExpData"]) - 1
        request = {"template_index": template_index, "model_index": 0,
                   "iterations": options["iterations"], "samples": options["covariance_samples"],
                   "terms": "D10 D50 D90", "weights": self.weights, "conditions": conditions}
        publish("imed_design", {"case": deepcopy(self.case), "method": method,
                                "survivors": survivors, "request": request})
        value, proposal, process = optimize_recipe(
            design_experiment, design_process, conditions, score,
            iterations=options["iterations"], cancelled=cancelled, workers=self.workers)
        stage = self._imed_stage(kind, target, pe_stage["candidates"])
        action = "d_optimal_design" if kind == "DO" else "model_discrimination"
        record_step(self.case, action, {"score": value, "experiment": proposal, "process": process,
                                       "request": request, "reason": reason, "stage": stage},
                    context="|".join(signature(self.case, index) for index in survivors))
        stage_path = save_native_case(stage["case_file"], self.case)
        report(f"Saved {stage['name']} ({method}) and its fitted models to {stage_path}.")
        publish("imed_result", {"case": deepcopy(self.case), "result": (value, proposal, process)})
        profiles = imed_recipe_profiles(proposal, method)
        if "stirring" not in profiles and "stirring_rate" not in self.setup["support_profiles"]:
            stirrer = design_experiment.get("CommandedProfiles", {}).get("stirring")
            if not stirrer and self.doe_decisions:
                stirrer = next(iter(self.doe_decisions["traces"].get("stirring", ())), None)
            if stirrer:
                from bo.decoder import fixed_horizon

                commanded = make_profile(stirrer["points"], stirrer["source"], step=stirrer["step"])
                profiles["stirring"] = fixed_horizon(commanded, profiles["temperature"].time[-1])
        if "InitialConcentration" in proposal:
            process["InitialConcentration"] = float(proposal["InitialConcentration"])
        return profiles, process, {"method": method, "design_reason": reason, "score": value,
                                   "stage_name": stage["name"], "stage_case": str(stage_path),
                                   "observations": len(self.case["ExpData"]),
                                   "surviving_models": [index + 1 for index in survivors],
                                   "aic_scores": {str(index + 1): score for index, score in scores.items()}}
