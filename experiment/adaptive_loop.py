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
from experiment.doe_campaign import ROLES, convert_profile
from experiment import loop_outputs
from experiment.loop_process import LOOP_WEIGHT_DEFAULTS, VALIDATION_PE_SETTINGS, starting_process_defaults
from experiment.models import Campaign, DeviceProfile, ExperimentDefinition, ProfilePoint
from experiment.recording import atomic_json
from hardware_devices.capabilities import is_monitor_only
from hardware_devices.limits import target_allowed, target_limits
from imed_python.campaign_import import ALIASES, CHANNELS, campaign_runs, import_run, read_json, seed_distribution
from imed_python.demo_campaign import compile_simulation, import_simulated_campaign, model_inputs, validation_model
from imed_python.design import discrimination_score, d_optimal_log_score, optimize_recipe, parameter_samples, recipe_space
from imed_python.estimation import FitObjective, fit_model
from imed_python.model_screening import matlab_screening_fields, pack_screening, screen_models
from imed_python.native_case import save_native_case
from imed_python.pe_settings import PE_WEIGHTS_KEY, PYFIT_DEFAULTS, PYFIT_SETTINGS_KEY, effective_pe_pbm
from imed_python.session import display_name, loop_stages, pack_fit, pe_signatures, record_step, signature
from imed_python.solver import pbm_backend_label
from imed_python.mixing import DEFAULT_MIXING, validate_mixing, average_mixing, trace_mean


MODES = (("External MODEL file", "external"), ("Bayesian optimization", "bo"), ("IMED", "imed"))
POSITIVE_PROCESS = ("SolvMass", "SolventDensity", "AntiSolventDensity", "SolidDensity", "ShapeFactor")
NONNEGATIVE_PROCESS = ("AntiSolvMass", "SeedMass", "PowerRatio")
DEVICE_RECORDING_COLUMNS = {
    "huber": ("temp_external", "temp_internal"), "opcua": ("value",), "mettler": ("weight_g",),
    "jkem": ("speed", "target"), "ismatec": ("flow", "target"), "masterflex": ("speed", "target"),
}


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
        "support_profiles": {}, "seed_pump": None,
        "imed": {"delta_aic": 2, "iterations": 5, "covariance_samples": 8,
                 "equilibration_s": 5, "conditions": {}},
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


def _merge(defaults, updates, label):
    if not isinstance(updates, dict) or set(updates) - set(defaults):
        raise ValueError(f"Check the fields in {label}.")
    return {**defaults, **updates}


def validate_setup(updates):
    defaults = default_setup()
    data = _merge(defaults, updates, "automatic loop settings")
    if not isinstance(data["reactor_unit"], str):
        raise ValueError("Select a model reactor unit.")
    if type(data["derive_initial_charges"]) is not bool:
        raise ValueError("Choose whether to derive starting charges from preparation pumps.")
    data["mixing"] = validate_mixing(data["mixing"])
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
        recipe_space(conditions)
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
    for key in POSITIVE_PROCESS:
        _number(values.get(key), key, 1e-12)
    for key in NONNEGATIVE_PROCESS:
        _number(values.get(key), key)
    return values


def _role_active(campaign, device_id, *, modeling_only=False):
    if campaign is None:
        return False
    for exp in campaign.experiments:
        for profile in exp.profiles:
            if profile.device_id != device_id:
                continue
            if not modeling_only or exp.model_window is None:
                if any(point.target != 0 for point in profile.points):
                    return True
                continue
            begin, end = exp.model_window
            times = np.array([point.time_s for point in profile.points])
            targets = np.array([point.target for point in profile.points])
            sample_times = np.union1d([begin, np.nextafter(end, begin)], times[(times >= begin) & (times < end)])
            values = (targets[np.clip(np.searchsorted(times, sample_times, side="right") - 1, 0, len(targets) - 1)]
                      if profile.interpolation == "step" else np.interp(sample_times, times, targets))
            if np.any(values != 0):
                return True
    return False


def device_configuration_changed(previous, current):
    fields = ("type", "connection", "defaults", "simulated", "target_limits", "pump_calibration", "connectivity")
    return previous is None or current is None or any(current.get(field) != previous.get(field) for field in fields)


def _column_exists(column, configs):
    device, _, field = column.rpartition(".")
    kind = configs.get(device, {}).get("type")
    return field in DEVICE_RECORDING_COLUMNS.get(kind, ())


def recording_mapping(setup, roles, configs):
    mapping = {key: next((column for column in ALIASES[key] if _column_exists(column, configs)), "")
               for key in CHANNELS}
    mapping["Temperature"] = f"{roles['T']}.temp_external"
    mapping.update(setup["channels"])
    invalid = [column for column in mapping.values() if column and not _column_exists(column, configs)]
    if invalid:
        raise ValueError("Map channels to assigned devices and monitors: " + ", ".join(invalid))
    columns = [column for column in mapping.values() if column]
    if len(set(columns)) != len(columns):
        raise ValueError("Map each recorded measurement to a distinct column.")
    return mapping


def read_completed_recordings(folder, expected, recording, mapping, processes_by_id, windows, *,
                              require_initial_concentration=True, roles=None, mixing=None):
    runs = campaign_runs(folder)
    run_ids = [read_json(file.with_name("run.json"))["experiment_id"] for _, _, file in runs]
    if set(run_ids) != set(expected) or len(set(run_ids)) != len(runs):
        raise ValueError("Recorded experiment identities differ from the completed campaign manifest.")
    demo = import_simulated_campaign(folder)
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
        start, end = windows.get(experiment_id, (recording["start_s"], recording["end_s"]))
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
            field = next((field for field in ("flow", "speed_rpm", "speed", "target")
                          if f"{device}.{field}" in headers and series(f"{device}.{field}").shape[1] >= 2), None)
            if not device or not field:
                raise ValueError("No usable AS scale. Assign an antisolvent pump with recorded delivery and a calibration/density.")
            if cfg.get("pump_calibration"):
                from experiment.pump_calibration_files import delivered_flow
                transform = lambda values: delivered_flow(cfg, values) / 3600.
            elif cfg.get("type") == "ismatec":
                density = _number(process.get("AntiSolventDensity"), "antisolvent density", 1e-12)
                transform = lambda values: values * density / 60.
            else:
                raise ValueError(f"Load a mass-flow calibration for antisolvent pump {device}, or map its reservoir scale.")
            feed_options = {"feed_channel": f"{device}.{field}", "feed_transform": transform}
            source = f"Calibrated pump delivery ({device}.{field})"
        experiment = import_run(
            file, mapping, start=start, end=end, initial_concentration=initial,
            concentration_multiplier=recording["concentration_multiplier"],
            scale_direction=recording["scale_direction"], smoothing=recording["smoothing"],
            seed_percentiles=recording["seed_percentiles"] if process["SeedMass"] else None,
            no_feed=recording["no_feed"], require_initial_concentration=require_initial_concentration,
            **feed_options,
        )
        experiment.update(Name=name, RecordingFile=str(file))
        experiment["CommandedProfiles"] = {key: value.payload() for key, value in campaign_profiles(experiment, process, payload, roles).items()}
        experiment["AntisolventSource"] = source
        definition = definitions.get(experiment_id, {})
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
        speed_source = "Assigned stirrer commands" if speed_field == "target" else "Measured stirring rate"
        if speed is not None:
            speed[0] -= origin
        else:
            entry = next((entry for entry in definition.get("profiles", []) if entry["device_id"] == stirrer), None)
            if entry:
                raw = np.asarray([[p["time_s"] for p in entry["points"]], [p["target"] for p in entry["points"]]])
                times = np.unique(np.r_[origin, raw[0, (raw[0] > origin) & (raw[0] < origin + duration)], origin + duration])
                speed = np.vstack((times - origin, np.interp(times, *raw)))
                speed_source = "Assigned stirrer commands"
        mean_rpm = trace_mean(speed, duration, "Stirring rate") if speed is not None else None
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


class RecipeCompiler:
    """Resolve process profiles to the assigned devices for LOOP and one-off execution."""

    def __init__(self, setup, template, configs, calibration_storage, assignments, monitors, mode):
        from experiment.run_stages import modeling_campaign
        self.setup, self.template, self.mode = setup, modeling_campaign(template, setup), mode
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
        self.demo = template.experiments[0].imed_simulation if template else None
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

    def check_support(self, campaign):
        support = self.setup["support_profiles"]
        bad_roles = set(support) - (set(self.roles) - {"T", "AS"})
        if bad_roles:
            raise ValueError("Support profiles need distinct assigned DoE roles: " + ", ".join(sorted(bad_roles)))
        for role, profile in support.items():
            self._support_profile(role, profile)
        if self.setup["seed_pump"]:
            if "seed" not in self.roles:
                raise ValueError("Assign the seed pump before enabling calibrated seed delivery.")
            from experiment.pump_calibration_files import seed_delivery_rate
            _number(seed_delivery_rate(self.configs[self.roles["seed"]], self.setup["seed_pump"]), "calibrated seed rate", 1e-12)
        if self.line_roles or (self.template and self.template.experiments[0].model_window is not None):
            return  # Timeline charging and support stages are carried into follow-up recipes.
        required = {role for role in self.roles if role not in {"T", "AS", "seed"}
                    and _role_active(campaign, self.roles[role])}
        if self.demo:
            required -= {"clean_sol", "valve_state"}
        missing = required - set(support) - ({"stirring_rate"} if self.mode == "bo" else set())
        if missing:
            raise ValueError("Define follow-up support profiles for: " + ", ".join(sorted(missing)))
        if self.setup["seed_pump"] and ("seed" not in self.roles or "seed" in support):
            raise ValueError("Assign a seed pump and choose either calibrated delivery or a support profile.")
        if self.setup["seed_pump"]:
            limits = target_limits(self.configs[self.roles["seed"]])
            if not target_allowed(self.setup["seed_pump"]["rpm"], limits):
                raise ValueError("Calibrated seed pump RPM exceeds the assigned pump limits.")
        if not self.demo and "seed" in self.roles and _role_active(campaign, self.roles["seed"]):
            if not self.setup["seed_pump"] and "seed" not in support:
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
        values = profile.values
        if role == "AS" and config["type"] == "ismatec" and not config.get("pump_calibration"):
            density = 1.
            if not self.demo and np.any(values):
                density = _number(process.get("AntiSolventDensity"), "antisolvent density (g/mL)", 1e-12)
            values, unit = values * 60 / density, "mL/min"
        else:
            mass_flow = self.role_specs[role].unit == "g/h"
            if role == "AS":
                values = values * 3600
            if mass_flow:
                from experiment.doe_campaign import ProfileAssignment
                semantic = self.line_roles.get(role, {}).get("role", role)
                conversion = ProfileAssignment(semantic, assignment.device_id, assignment.density_g_ml)
                values, unit, _ = convert_profile(self.calibration_storage, conversion, config, values)
            else:
                unit = target_limits(config)["unit"]
        return DeviceProfile(assignment.device_id, tuple(ProfilePoint(float(t), float(v))
                             for t, v in zip(profile.time, values)),
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
            if "AS" in self.roles:
                device_profiles["AS"] = self.device_profile("AS", feed, process)
        stirring = recipe.get("stirring")
        support = self.setup["support_profiles"]
        if stirring is not None and "stirring_rate" in support:
            raise ValueError("The BO stirring recommendation conflicts with the configured main stirrer support profile.")
        if stirring is not None:
            if "stirring_rate" not in self.roles and np.any(stirring.values != 0):
                raise ValueError("Assign a main stirrer to execute the BO stirring recommendation.")
        if stirring is not None and "stirring_rate" in self.roles:
            device_profiles["stirring_rate"] = self.device_profile("stirring_rate", stirring)
        if ("stirring_rate" in self.roles and _role_active(self.template, self.roles["stirring_rate"])
                and "stirring_rate" not in device_profiles and "stirring_rate" not in support and not self.line_roles
                and not (self.template and self.template.experiments[0].model_window is not None)):
            raise ValueError("Provide a recorded or configured main stirrer profile for the follow-up.")
        seed_mass = _number(process.get("SeedMass", default_seed), "seed mass")
        if recipe.get("volume") is not None:
            requested = float(recipe["volume"].values[0])
            current = process.get("InitialVolume_ml")
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
            calibration = self.setup["seed_pump"]
            if modeling and seed_mass > 0 and not self.setup["recording"]["seed_percentiles"]:
                raise ValueError("Configure seed percentiles before executing a seeded follow-up.")
            if seed_mass and calibration:
                from experiment.pump_calibration_files import seed_delivery_rate
                end = seed_mass / seed_delivery_rate(self.configs[self.roles["seed"]], calibration)
                if end >= duration:
                    raise ValueError("Seed addition exceeds the recommended experiment duration.")
                seed = Profile(np.array([0., end, duration]), np.array([calibration["rpm"], 0., 0.]),
                               "Calibrated seed dose", step=True)
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
        if original.model_window is None:
            raise ValueError("Mark the Experiment and Equilibration stages in Experiment Definition before running a modeling loop.")
        if process.get("SeedMass", default_seed) != default_seed:
            raise ValueError("Keep the timeline's fixed seed mass for follow-ups, or define calibrated seed delivery.")
        begin, end = original.model_window
        volume_scale = 1.
        if "RequestedInitialVolume_ml" in process:
            from experiment.material_balance import initial_charges
            inferred = initial_charges(original, process, self.roles, self.configs, line_roles=self.line_roles)
            reference = inferred.get("InitialVolume_ml", process.get("InitialVolume_ml"))
            if reference is None or reference <= 0:
                raise ValueError("Volume decisions require a known starting liquid volume.")
            volume_scale = process["RequestedInitialVolume_ml"] / reference
            if not np.isclose(volume_scale, 1.):
                charge_devices = {device for role, device in self.roles.items()
                                  if self.line_roles.get(role, {}).get("role", role) in {"AS", "solvent", "solution"}}
                adjustable = [p for p in original.profiles if p.device_id in charge_devices
                              and any(point.target > 0 for point in p.points if point.time_s < begin)]
                if not adjustable:
                    raise ValueError("Initial-volume decisions need defined preparation charging pumps. Add those stages or hold volume fixed in BO.")
                if not process.get("VolumeFromDesignConditions"):
                    process["SolvMass"] *= volume_scale
                    process["AntiSolvMass"] *= volume_scale
                process["InitialVolume_ml"] = process["RequestedInitialVolume_ml"]
        shift = begin + duration - end
        by_device = {profile.device_id: profile for profile in modeled.values()}
        profiles = []
        for profile in original.profiles:
            if not np.isclose(volume_scale, 1.) and profile.device_id in charge_devices:
                from experiment.pump_calibration_files import delivered_flow, convert_flow
                cfg = self.configs[profile.device_id]
                def scaled(target):
                    if cfg.get("pump_calibration"):
                        return float(convert_flow(cfg["pump_calibration"], cfg["type"], delivered_flow(cfg, [target]) * volume_scale)[0])
                    if cfg["type"] == "ismatec":
                        return target * volume_scale
                    raise ValueError(f"Load the charging calibration for {profile.device_id} before varying initial volume.")
                profile = DeviceProfile(profile.device_id,
                    tuple(ProfilePoint(p.time_s, scaled(p.target) if p.time_s < begin else p.target) for p in profile.points),
                    profile.interpolation, profile.unit)
            replacement = by_device.get(profile.device_id)
            if replacement is None:
                if any(begin + duration < point.time_s < end for point in profile.points):
                    raise ValueError(f"The recommended duration is shorter than {profile.device_id}'s support sequence.")
                points = tuple(ProfilePoint(p.time_s + shift if p.time_s >= end else p.time_s, p.target)
                               for p in profile.points)
            else:
                before = tuple(p for p in profile.points if p.time_s < begin)
                middle = tuple(ProfilePoint(begin + p.time_s, p.target) for p in replacement.points)
                after = tuple(ProfilePoint(p.time_s + shift, p.target) for p in profile.points if p.time_s > end)
                points = (*before, *middle, *after)
            profiles.append(DeviceProfile(profile.device_id, points, profile.interpolation, profile.unit))
        modes = tuple(replace(mode, time_s=mode.time_s + shift if mode.time_s >= end else mode.time_s)
                      for mode in original.control_modes)
        from experiment.run_stages import RunStage
        stages = tuple(stage for stage in original.run_stages if stage.end_s <= begin + 1e-7)
        stages += (RunStage("modeled-experiment", f"{self.mode.upper()} proposed experiment", "Experiment", begin, begin + duration),)
        stages += tuple(replace(stage, start_s=stage.start_s + shift, end_s=stage.end_s + shift)
                        for stage in original.run_stages if stage.start_s >= end - 1e-7)
        experiment = ExperimentDefinition(name, tuple(profiles), source=source, control_modes=modes,
                                          model_window=(begin, begin + duration), run_stages=stages,
                                          initial_mass_g=original.initial_mass_g)
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
                 *, bo_settings=None, size_term, workers=1):
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
        self.size_term, self.workers = size_term, workers
        self.followups = 0
        self.process_by_id = {}
        self.windows = {}
        self.used_recordings = set()
        self.recipe_origins = {}
        unmapped = {profile.device_id for experiment in campaign.experiments for profile in experiment.profiles}
        unmapped -= set(self.roles.values())
        if unmapped:
            raise ValueError("Assign the initial actuators to known profile roles: " + ", ".join(sorted(unmapped)))
        self.check_devices(campaign, configs, connected)
        self._prepare_measurements(campaign)
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
        self.compiler.check_support(campaign)

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
        self.process = _process(self.compiler.process_defaults(self.case))
        if self.setup["recording"]["initial_concentration"] is not None:
            self.process.setdefault("InitialConcentration", self.setup["recording"]["initial_concentration"])
        prepared = []
        for experiment in campaign.experiments:
            process = _process({**self.process, **self.setup["initial_processes"].get(experiment.name, {})})
            if self.setup["derive_initial_charges"]:
                from experiment.material_balance import initial_charges
                process = _process(initial_charges(experiment, process, self.roles, self.configs,
                                   overrides=self.setup["initial_processes"].get(experiment.name, {}),
                                   line_roles=self.compiler.line_roles))
            if "InitialConcentration" in process:
                _number(process["InitialConcentration"], "initial concentration", 0, 1 - 1e-12)
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
            prepared.append(replace(experiment, initial_mass_g=initial))
            if experiment.model_window is not None:
                self.windows[experiment.experiment_id] = experiment.model_window
        self.template = replace(campaign, experiments=tuple(prepared))
        self.compiler.template = self.template
        self.mapping = recording_mapping(self.setup, self.roles, self.configs)
        self._require_channels("Temperature")
        recording = self.setup["recording"]
        if recording["initial_concentration"] is None:
            self._require_channels("Concentration")
        if not recording["no_feed"]:
            if not self.mapping.get("AS scale"):
                if "AS" not in self.roles:
                    raise ValueError("Assign an antisolvent pump or map an AS scale for addition tracking.")
                cfg = self.configs[self.roles["AS"]]
                if cfg["type"] != "ismatec" and not cfg.get("pump_calibration"):
                    raise ValueError("Without an AS scale, load the antisolvent pump's mass-flow calibration.")
        elif "AS" in self.roles and _role_active(campaign, self.roles["AS"], modeling_only=True):
            raise ValueError("Clear no_feed or remove antisolvent delivery from the selected modeling period.")
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
        if not self.case.get("Solubility") or not self.case.get("ModelParameter"):
            raise ValueError("Load an IMED case with solubility and candidate kinetic models.")
        for index, model in enumerate(self.case["ModelParameter"], 1):
            table = np.asarray(model.get("paraset", []), dtype=float)
            if table.shape != (35, 4):
                raise ValueError(f"Set valid parameter bounds and enable estimation in IMED model {index}.")
            active = table[table[:, 0] == 1, 1:]
            if (not len(active) or not np.isfinite(active).all() or np.isnan(table).any()
                    or np.isposinf(table).any() or np.any(active[:, 1] > active[:, 2])):
                raise ValueError(f"Set valid parameter bounds and enable estimation in IMED model {index}.")
        weights = self.case.get(PE_WEIGHTS_KEY, self.case.get("objweightfactor", {}))
        self.weights = [float(weights.get(f"W{i}", default)) for i, default in enumerate(LOOP_WEIGHT_DEFAULTS, 1)]
        if not np.isfinite(self.weights).all() or min(self.weights) < 0 or sum(w > 0 for w in self.weights) < 2:
            raise ValueError("Enable at least two IMED response weights so the loop can reach D-optimal design.")
        if self.weights[0]:
            self._require_channels("FBRM")
        if self.weights[1]:
            self._require_channels("Concentration")
        if self.weights[2]:
            self._require_channels("D10", "D50", "D90")
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
        conditions = options["conditions"]
        if not conditions:
            raise ValueError("Enter automatic IMED design conditions in Setup.")
        if self.demo and options["equilibration_s"] <= 0:
            raise ValueError("Add a positive equilibration duration for simulated IMED follow-ups.")
        design_settings = self.case.get("pbmsetting", {})
        self.case["pbmsetting"] = {**self.pe_mesh, **(design_settings if isinstance(design_settings, dict) else {})}
        addition, bounds = recipe_space(conditions)
        temperature_limits = target_limits(self.configs[self.roles["T"]])
        if not all(target_allowed(temp, temperature_limits) for temp in conditions["temperature"]):
            raise ValueError("IMED design temperatures exceed the assigned temperature controller limits.")
        if self.setup["recording"]["no_feed"] and addition:
            raise ValueError("IMED design adds antisolvent but recordings are configured as no_feed.")
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
            if self.setup["seed_pump"] is None and list(conditions["seed_bounds"]) != [self.process["SeedMass"]] * 2:
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

    def _read_batch(self, manifest, target, report):
        for experiment_id in manifest["experiments"]:
            if window := self.windows.get(experiment_id):
                report(f"Importing only the modeling experiment period {window[0]:g}–{window[1]:g} s; "
                       "full preparation and cleanup recordings are retained.")
        experiments, processes, runs = read_completed_recordings(
            Path(manifest["campaign_folder"]), manifest["experiments"], self.setup["recording"],
            self.mapping, self.process_by_id, self.windows, roles=self.roles,
            mixing=self.setup["mixing"] if self.mode == "imed" and not self.demo else None)
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
            self.case.setdefault("PythonWorkflow", {})["initial_input_groups"] = groups
            if self.bo is not None:
                self.bo["initial_input_groups"] = groups
            if not self.demo:
                self.process.update(processes[0])
            progress(f"Initial campaign operating inputs available for BO / SA: {', '.join(groups) or 'stirring and volume fixed'}.")
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
        _process(process)
        campaign, window = self.compiler.build(
            profiles, process, name=f"{self.mode.upper()} follow-up {self.followups + 1}",
            source=self.mode, default_seed=self.process["SeedMass"], modeling=True)
        experiment = campaign.experiments[0]
        self.process_by_id[experiment.experiment_id] = process
        self.windows[experiment.experiment_id] = window
        process_state = {key: float(process[key]) for key in (*POSITIVE_PROCESS, *NONNEGATIVE_PROCESS)}
        if (initial := process.get("InitialConcentration")) is not None:
            process_state["InitialConcentration"] = float(initial)
        payload = {"format": "crystaloop-recommendation", "version": 1, "backend": self.mode,
                   "input_campaign_id": manifest["campaign_id"], "campaign": asdict(campaign),
                   "process": process_state, "crystallization_window_s": self.windows[experiment.experiment_id],
                   **details}
        if self.mode == "imed":
            save_native_case(target / "fit.imed.json", self.case)
            self.active_models = survivors
        self.recipe_origins[campaign.campaign_id] = {
            "RecipeSource": details["method"], "RecipeReason": details.get("design_reason", ""),
            "RecipeStage": details["stage_name"], "RecipeModelFolder": str(target),
        }
        loop_outputs.modeled_recipe(target, profiles, campaign, details.get("predictions"))
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
        result = propose(state, report=report)
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
            score = partial(d_optimal_log_score, model, solubility=solubility, settings=settings,
                            enabled_terms=[weight > 0 for weight in self.weights], samples=samples)
        if len(survivors) > 1:
            report(f"IMED selected {method}. {reason}")
        options = self.setup["imed"]
        conditions = {**options["conditions"], "equilibration_s": options["equilibration_s"]}
        design_experiment = {**experiments[-1]}
        if conditions["seed_bounds"][1] > 0:
            design_experiment["InitialCSDsData"] = (model_inputs(True)[1]["InitialCSDsData"] if self.demo
                                                    else seed_distribution(self.setup["recording"]["seed_percentiles"]))
        request = {"template_index": len(self.case["ExpData"]) - 1, "model_index": 0,
                   "iterations": options["iterations"], "samples": options["covariance_samples"],
                   "terms": "D10 D50 D90", "weights": self.weights, "conditions": conditions}
        publish("imed_design", {"case": deepcopy(self.case), "method": method,
                                "survivors": survivors, "request": request})
        value, proposal, process = optimize_recipe(
            design_experiment, processes[-1], conditions, score,
            iterations=options["iterations"], cancelled=cancelled, workers=self.workers)
        stage = self._imed_stage(kind, target, pe_stage["candidates"])
        action = "d_optimal_design" if kind == "DO" else "model_discrimination"
        record_step(self.case, action, {"score": value, "experiment": proposal, "process": process,
                                       "request": request, "reason": reason, "stage": stage},
                    context="|".join(signature(self.case, index) for index in survivors))
        stage_path = save_native_case(stage["case_file"], self.case)
        report(f"Saved {stage['name']} ({method}) and its fitted models to {stage_path}.")
        publish("imed_result", {"case": deepcopy(self.case), "result": (value, proposal, process)})
        profiles = {"temperature": Profile(*proposal["TemperatureData"], method),
                    "feed": Profile(*proposal["FlowRateData"], method)}
        process["InitialConcentration"] = options["conditions"]["concentration"]
        return profiles, process, {"method": method, "design_reason": reason, "score": value,
                                   "stage_name": stage["name"], "stage_case": str(stage_path),
                                   "observations": len(self.case["ExpData"]),
                                   "surviving_models": [index + 1 for index in survivors],
                                   "aic_scores": {str(index + 1): score for index, score in scores.items()}}
