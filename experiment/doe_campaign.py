"""Compile generated process profiles into assigned, executable campaigns."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json

import numpy as np
import pandas as pd

from experiment.models import Campaign, ControlModePoint, DeviceProfile, ExperimentDefinition, ProfilePoint
from hardware_devices.capabilities import is_monitor_only
from hardware_devices.limits import target_limits


@dataclass(frozen=True)
class ProfileRole:
    label: str
    unit: str
    compatible_types: tuple[str, ...]


ROLES = {
    "T": ProfileRole("Reactor temperature", "°C", ("huber",)),
    "AS": ProfileRole("Antisolvent", "g/h", ("masterflex", "ismatec")),
    "solution": ProfileRole("Solution", "g/h", ("masterflex", "ismatec")),
    "solvent": ProfileRole("Solvent", "g/h", ("masterflex", "ismatec")),
    "clean_sol": ProfileRole("Cleaning solvent", "g/h", ("masterflex", "ismatec")),
    "seed": ProfileRole("Seed pump", "RPM", ("masterflex",)),
    "stirring_rate": ProfileRole("Main stirrer", "RPM", ("jkem",)),
    "stirring_rate_seed": ProfileRole("Seed stirrer", "RPM", ("jkem",)),
    "stirring_rate_sol": ProfileRole("Solution stirrer", "RPM", ("jkem",)),
    "valve_state": ProfileRole("Outlet valve", "binary", ("nidaq_valve",)),
}

DEFAULT_ASSIGNMENTS = {
    "T": "TEM_MAIN", "AS": "PMP_AS", "solution": "PMP_SOLU", "solvent": "PMP_SOLV",
    "clean_sol": "PMP_CLEN", "seed": "PMP_SEED", "stirring_rate": "STR_MAIN",
    "stirring_rate_seed": "STR_SEED", "stirring_rate_sol": "STR_SOLU", "valve_state": "VALVE",
}


@dataclass(frozen=True)
class ProfileAssignment:
    role: str
    device_id: str
    density_g_ml: float | None = None


def convert_profile(storage, assignment, config, values):
    role = ROLES[assignment.role]
    if config["type"] not in role.compatible_types or is_monitor_only(config):
        raise ValueError(f"{role.label} cannot be assigned to {config['type']}")
    if role.unit != "g/h":
        return values, role.unit, "identity"
    if config.get("pump_calibration"):
        from experiment.pump_calibration_files import convert_flow
        calibration = config["pump_calibration"]
        unit = calibration["output_unit"]
        if unit != target_limits(config)["unit"]:
            raise ValueError("Review this Ismatec calibration in Define pump profile and confirm the recorded command units.")
        return (convert_flow(calibration, config["type"], values), unit,
                f"Loaded calibration: {calibration['source_name']} (g/h → {unit})")
    if config["type"] == "ismatec":
        if config.get("simulated"):
            return values / 60, "RPM", "Simulated delivery (g/min → virtual RPM)"
        if np.any(values):
            raise ValueError("Load the Ismatec antisolvent mass-flow calibration in RPM before generating pump commands.")
        return values, "RPM", "Pump stays off"
    converter_names = {"AS": "as_rate_converter", "solution": "solution_rate_converter",
                       "solvent": "solvent_rate_converter", "clean_sol": "cleaning_sol_rate_converter"}
    converter = getattr(storage, converter_names[assignment.role])
    # Preserve sign and exact zero; the fitted calibration is for magnitude.
    converted = np.sign(values) * converter.convert_to_rpm(np.abs(values))
    if not np.all(np.isfinite(converted)) or np.any(converted * values < 0):
        raise ValueError(f"{role.label}: calibration produced invalid pump targets")
    return converted, "RPM", f"{assignment.role} pump calibration (g/h → RPM)"


def sampled_doe_factors(creator):
    factors = set(getattr(creator, "DoE_val", {}))
    snapshots = getattr(creator, "parameter_snapshots", {}).values()
    if len({snapshot.get("ARR") for snapshot in snapshots} - {None}) > 1:
        factors.add("ARR")
    return tuple(sorted(factors))


class DoeCampaignService:
    def __init__(self, creator):
        self.creator = creator
        self.roles = getattr(creator, "line_roles", ROLES)
        self.recipe_lanes = {lane["id"]: lane for lane in getattr(creator, "recipe", {}).get("lanes", [])}
        self.frames = {key: frame.copy(deep=True) for key, frame in creator.exp_storage.data.items()}
        if not self.frames:
            raise ValueError("Generate a DoE first")
        self.imed_simulations = {
            name: options["imed_simulation"]
            for name, options in getattr(creator, "parameter_snapshots", {}).items()
            if name in self.frames and "imed_simulation" in options
        }
        if self.imed_simulations and len(self.imed_simulations) != len(self.frames):
            raise ValueError("Keep IMED validation runs in a separate DoE preset.")
        # Match the generated DoE plots/export's existing return-pump segments,
        # but keep source flow units until the assigned driver is known.
        self.return_ramps_applied = (not self.imed_simulations and not self.recipe_lanes
                                     and hasattr(creator.exp_storage, "add_backwards_ramp"))
        self.nominal_frames = self.frames
        if self.return_ramps_applied:
            self.frames = {key: creator.exp_storage.add_backwards_ramp(
                frame, creator.exp_storage.COLUMNS_W_BACKPROFILE
            ) for key, frame in self.frames.items()}

    def required_roles(self):
        if self.recipe_lanes:
            return set(self.recipe_lanes)
        return {role for role in self.roles if any(
            role in frame and (role == "T" or np.any(frame[role].to_numpy(dtype=float) != 0))
            for frame in self.frames.values()
        )}

    def compile(self, assignments, configs, monitor_device_ids=(), *, imed_monitors=None):
        assignments = tuple(assignments)
        roles = [item.role for item in assignments]
        device_ids = [item.device_id for item in assignments]
        if len(roles) != len(set(roles)) or len(device_ids) != len(set(device_ids)):
            raise ValueError("Assign each profile once and use a different device for each profile")
        if any(role not in self.roles for role in roles):
            raise ValueError("Unknown generated profile role")
        missing = self.required_roles() - set(roles)
        if missing:
            raise ValueError("Assign active profiles: " + ", ".join(self.roles[key].label for key in sorted(missing)))
        if not assignments:
            raise ValueError("Assign at least one generated profile")
        for device_id in device_ids:
            if device_id not in configs:
                raise ValueError(f"Assigned device no longer exists: {device_id}")
        for device_id in monitor_device_ids:
            if device_id not in configs or not is_monitor_only(configs[device_id]):
                raise ValueError(f"Monitor must be a read-only device: {device_id}")
        role_devices = dict(zip(roles, device_ids))
        frames = self.frames
        if self.return_ramps_applied and any("pump_corrections" in configs[device] for device in device_ids):
            legacy_columns = [column for column in self.creator.exp_storage.COLUMNS_W_BACKPROFILE
                              if column not in role_devices or "pump_corrections" not in configs[role_devices[column]]]
            frames = {name: self.creator.exp_storage.add_backwards_ramp(frame.copy(), legacy_columns)
                      for name, frame in self.nominal_frames.items()}
        experiments = []
        for name, frame in frames.items():
            # Disabled zero-duration generator sections can repeat an identical
            # boundary row. Remove only exact duplicates; conflicting targets
            # at the same instant remain an error.
            frame = frame.drop_duplicates()
            if "t" not in frame or frame.empty:
                raise ValueError(f"{name}: missing generated timeline")
            times = frame["t"].to_numpy(dtype=float) * 3600
            if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0):
                raise ValueError(f"{name}: times must be finite, nonnegative and strictly increasing")
            snapshot = self.creator.parameter_snapshots.get(name, {})
            seed = snapshot.get("seed")
            planned_seed = float(seed["value"]) if isinstance(seed, dict) and snapshot.get("if_seed") else None
            if snapshot.get("if_seed") is False:
                planned_seed = 0.
            seed_doses = snapshot.get("seed_doses", ())
            if seed_doses:
                planned_seed = sum(dose["mass_g"] for dose in seed_doses)
            seed_config = configs.get(role_devices.get("seed"), {})
            if not self.recipe_lanes and planned_seed and seed_config.get("seed_calibration"):
                from experiment.pump_calibration_files import seed_dose_command

                rpm, seconds = seed_dose_command(seed_config, None, planned_seed)
                positive = np.flatnonzero(frame["seed"].to_numpy() > 0)
                if positive.size < 2:
                    raise ValueError(f"{name}: the legacy seed profile needs a positive pumping interval.")
                start, stop = times[positive[[0, -1]]]
                times = np.interp(np.minimum(times, stop), [0., start, stop], [0., start, start + seconds]) + np.maximum(times - stop, 0.)
                frame = frame.assign(seed=np.sign(frame["seed"]) * rpm)
            profiles = []
            for assignment in assignments:
                if assignment.role not in frame:
                    raise ValueError(f"{name}: missing {assignment.role} profile")
                values = frame[assignment.role].to_numpy(dtype=float)
                semantic = self.recipe_lanes.get(assignment.role, {}).get("role", assignment.role)
                conversion = ProfileAssignment(semantic, assignment.device_id, assignment.density_g_ml)
                config = configs[assignment.device_id]
                profile_times = times
                if ROLES[semantic].unit == "g/h" and config.get("pump_calibration"):
                    from experiment.pump_calibration_files import calibration_knots

                    profile_times, values = calibration_knots(config, profile_times, values)
                converted, unit, _ = convert_profile(self.creator.exp_storage, conversion, config, values)
                if profile_times[0] > 0:
                    profile_times = np.insert(profile_times, 0, 0.)
                    converted = np.insert(converted, 0, converted[0])
                profile = DeviceProfile(
                    assignment.device_id,
                    tuple(ProfilePoint(float(t), float(v)) for t, v in zip(profile_times, converted)),
                    "step" if semantic == "valve_state" else "linear", unit,
                )
                if semantic == "seed" and seed_doses:
                    doses = [dose for dose in seed_doses if dose["lane"] == assignment.role and dose["mass_g"] > 0]
                    if doses:
                        profile = calibrated_doses(profile, doses, config, snapshot.get("model_window"))
                liquid_doses = [dose for dose in snapshot.get("liquid_doses", ())
                                if dose["lane"] == assignment.role and dose["mass_g"] > 0]
                if liquid_doses:
                    profile = calibrated_doses(profile, liquid_doses, config)
                profiles.append(profile)
            simulation = self.imed_simulations.get(name)
            if simulation is not None:
                from imed_python.demo_campaign import compile_simulation
                simulation = compile_simulation(simulation, role_devices, configs,
                                                monitor_device_ids, imed_monitors, profiles)
            stages = snapshot.get("run_stages", ())
            if not stages and self.recipe_lanes:
                from experiment.run_stages import recipe_run_stages
                # Older exact snapshots already saved sampled stage durations.
                stages = recipe_run_stages(self.creator.recipe,
                                          {key: entry["value"] for key, entry in snapshot.items()
                                           if key.endswith(".duration")})
            modes = tuple(ControlModePoint(item["time_s"], role_devices[item["lane"]], item["mode"])
                          for item in snapshot.get("control_modes", []) if item["lane"] in role_devices)
            definition = ExperimentDefinition(name, tuple(profiles), source="doe", imed_simulation=simulation,
                                              control_modes=modes, model_window=snapshot.get("model_window"),
                                              run_stages=stages, seed_mass_g=planned_seed)
            # The shared validator applies the currently configured device limits.
            ExperimentDefinition.from_frame(definition.to_frame(), configs)
            experiments.append(definition)
        factors = sampled_doe_factors(self.creator)
        return Campaign("Generated DoE", tuple(experiments), monitor_device_ids=tuple(monitor_device_ids),
                        doe_factors=(factors or None) if self.imed_simulations else factors,
                        line_roles={key: {**lane, "device_id": role_devices[key]} for key, lane in self.recipe_lanes.items()})

    def assignment_details(self, assignments, configs):
        result = []
        for assignment in assignments:
            semantic = self.recipe_lanes.get(assignment.role, {}).get("role", assignment.role)
            original = ProfileAssignment(semantic, assignment.device_id, assignment.density_g_ml)
            _, unit, conversion = convert_profile(self.creator.exp_storage, original, configs[assignment.device_id], np.array([0.]))
            result.append({"role": assignment.role, "device_id": assignment.device_id,
                           "source_unit": self.roles[assignment.role].unit, "target_unit": unit,
                           "density_g_ml": 1.0 if configs[assignment.device_id]["type"] == "ismatec" else None, "conversion": conversion})
            custom = configs[assignment.device_id].get("pump_calibration")
            if self.roles[assignment.role].unit == "g/h" and custom:
                result[-1].update(calibration_file=custom["source_name"], calibration_sha256=custom["sha256"],
                                  calibration_points=custom["points"], density_g_ml=None)
            elif self.roles[assignment.role].unit == "g/h" and configs[assignment.device_id]["type"] == "masterflex":
                converter_name = {"AS": "as_rate_converter", "solution": "solution_rate_converter",
                                  "solvent": "solvent_rate_converter", "clean_sol": "cleaning_sol_rate_converter"}[semantic]
                converter = getattr(self.creator.exp_storage, converter_name)
                result[-1]["calibration_coefficients"] = list(map(float, converter.res.x))
                result[-1]["calibration_file"] = Path(converter.folder_loc).name
        return result

    def parameter_table(self):
        rows = []
        snapshots = getattr(self.creator, "parameter_snapshots", {})
        for name in self.frames:
            row = {"experiment": name}
            for key, value in snapshots.get(name, {}).items():
                if isinstance(value, dict) and "value" in value:
                    row[key] = value["value"]
                elif isinstance(value, (str, bool, int, float)):
                    row[key] = value
            if name in self.creator.exp_storage.DoE:
                for key, values in self.creator.exp_storage.DoE[name].items():
                    if len(values):
                        row[key] = values.iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)

    def export_csv(self, parent, campaign=None, assignments=(), configs=None):
        """Create a new export directory; never overwrite an earlier export."""
        directory = Path(parent) / f"doe_{uuid4().hex[:12]}"
        directory.mkdir()
        try:
            self.parameter_table().to_csv(directory / "parameters.csv", index=False)
            raw_rows = []
            for name, frame in self.frames.items():
                for role, spec in self.roles.items():
                    if role in frame:
                        raw_rows.extend({"experiment": name, "time_s": float(t) * 3600,
                                         "profile_role": role, "target": float(value), "unit": spec.unit}
                                        for t, value in zip(frame["t"], frame[role]))
            wide = []
            for name, frame in self.frames.items():
                output = frame.rename(columns={"t": "time_s"}).copy()
                output["time_s"] *= 3600
                output.insert(0, "experiment", name)
                wide.append(output)
                output.drop(columns="experiment").to_csv(directory / f"{name}_generated.csv", index=False)
            pd.concat(wide, ignore_index=True).to_csv(directory / "generated_profiles.csv", index=False)
            metadata = {"schema_version": 1, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                        "time_unit": "s", "source_time_unit": "h", "strategy": self.creator.TYPE,
                        "kind": "planned_profiles", "experiment_count": len(self.frames)}
            metadata["return_pump_segments"] = self.return_ramps_applied
            if campaign is not None:
                from experiment.pump_corrections import corrected_campaign

                campaign = corrected_campaign(campaign, configs or {})
                assigned_rows = []
                for experiment in campaign.experiments:
                    for profile in experiment.profiles:
                        assigned_rows.extend({"campaign_id": campaign.campaign_id,
                                              "experiment_id": experiment.experiment_id, "experiment": experiment.name,
                                              "device_id": profile.device_id, "time_s": point.time_s,
                                              "target": point.target, "unit": profile.unit,
                                              "interpolation": profile.interpolation}
                                             for point in profile.points)
                wide = []
                for experiment in campaign.experiments:
                    output = experiment.to_frame()
                    output.to_csv(directory / f"{experiment.name}_assigned.csv", index=False)
                    output.insert(0, "experiment", experiment.name)
                    output.insert(0, "experiment_id", experiment.experiment_id)
                    output.insert(0, "campaign_id", campaign.campaign_id)
                    wide.append(output)
                pd.concat(wide, ignore_index=True).to_csv(directory / "assigned_profiles.csv", index=False)
                details = self.assignment_details(assignments, configs or {})
                pd.DataFrame(details).to_csv(directory / "assignments.csv", index=False)
                metadata.update(campaign_id=campaign.campaign_id, monitor_device_ids=campaign.monitor_device_ids,
                                assignments=details, devices=configs)
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            # Keep partial output visibly incomplete instead of pretending it succeeded.
            (directory / "EXPORT_FAILED.txt").write_text("Export did not complete. Do not use this directory as a complete campaign.", encoding="utf-8")
            raise
        return directory


def calibrated_doses(profile, doses, config, window=None):
    """Insert device-unit pulses into the allotted sections of a pump profile."""
    from experiment.pump_calibration_files import seed_dose_command
    from experiment.pump_profile_data import dose_duration

    points = {point.time_s: point.target for point in profile.points}
    for dose in doses:
        start = dose["start_s"]
        if setting := dose.get("setting"):
            target = setting["target"]
            if setting["unit"] != profile.unit:
                raise ValueError(f"{dose['name']}: the selected pump setting uses {setting['unit']}. The assigned pump uses {profile.unit}.")
            seconds, rate = dose_duration(config, target, dose["mass_g"])
            if rate > dose["limit_g_h"] + 1e-9:
                raise ValueError(f"{dose['name']}: {target:g} {profile.unit} exceeds the System settings delivery rate limit.")
        else:
            if not config.get("seed_calibration"):
                raise ValueError(f"Load a dry-seed dose calibration for {profile.device_id} to convert the planned grams.")
            target, seconds = seed_dose_command(config, None, dose["mass_g"])
        end = start + seconds
        limit = min(dose["end_s"], window[0]) if window else dose["end_s"]
        if end > limit + 1e-7:
            raise ValueError(f"{dose['name']}: {dose['mass_g']:g} g needs {seconds:g} s at {target:g} {profile.unit}, "
                             f"but the dosing stage has {limit - start:g} s available.")
        # Round-off in the generated hours grid can land a boundary on either side of start.
        restore = next((value for time, value in points.items() if abs(time - end) < 1e-8), 0.)
        edge = min(1e-6, seconds / 4., start / 2.)
        if edge:
            times = sorted(points)
            before = float(np.interp(start - edge, times, [points[time] for time in times]))
        points = {time: value for time, value in points.items() if not start - 1e-8 <= time <= end + 1e-8}
        if edge:
            points[start - edge] = before
        points.update({start: target, end - min(1e-6, seconds / 4.): target, end: restore})
    return DeviceProfile(profile.device_id, tuple(ProfilePoint(t, v) for t, v in sorted(points.items())),
                         profile.interpolation, profile.unit)
