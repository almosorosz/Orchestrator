"""Compile generated process profiles into assigned, executable campaigns."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import math

import numpy as np
import pandas as pd

from experiment.models import Campaign, DeviceProfile, ExperimentDefinition, ProfilePoint
from hardware_devices.capabilities import is_monitor_only


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


class DoeCampaignService:
    def __init__(self, creator):
        self.creator = creator
        self.frames = {key: frame.copy(deep=True) for key, frame in creator.exp_storage.data.items()}
        if not self.frames:
            raise ValueError("Generate a DoE first")
        # Match the generated DoE plots/export's existing return-pump segments,
        # but keep source flow units until the assigned driver is known.
        self.return_ramps_applied = hasattr(creator.exp_storage, "add_backwards_ramp")
        if self.return_ramps_applied:
            self.frames = {key: creator.exp_storage.add_backwards_ramp(
                frame, creator.exp_storage.COLUMNS_W_BACKPROFILE
            ) for key, frame in self.frames.items()}

    def required_roles(self):
        return {role for role in ROLES if any(
            role in frame and (role == "T" or np.any(frame[role].to_numpy(dtype=float) != 0))
            for frame in self.frames.values()
        )}

    def _convert(self, assignment, config, values):
        role = ROLES[assignment.role]
        if config["type"] not in role.compatible_types or is_monitor_only(config):
            raise ValueError(f"{role.label} cannot be assigned to {config['type']}")
        if role.unit != "g/h":
            return values, role.unit, "identity"
        if config.get("pump_calibration"):
            from experiment.pump_calibration_files import convert_flow, OUTPUT_UNITS
            calibration = config["pump_calibration"]
            return (convert_flow(calibration, config["type"], values), OUTPUT_UNITS[config["type"]],
                    f"Loaded calibration: {calibration['source_name']} (g/h → {OUTPUT_UNITS[config['type']]})")
        if config["type"] == "ismatec":
            return values / 60, "mL/min", "g/h ÷ 60 (assumed density 1 g/mL)"
        converter_names = {"AS": "as_rate_converter", "solution": "solution_rate_converter",
                           "solvent": "solvent_rate_converter", "clean_sol": "cleaning_sol_rate_converter"}
        converter = getattr(self.creator.exp_storage, converter_names[assignment.role])
        # Preserve sign and exact zero; the fitted calibration is for magnitude.
        converted = np.sign(values) * converter.convert_to_rpm(np.abs(values))
        if not np.all(np.isfinite(converted)) or np.any(converted * values < 0):
            raise ValueError(f"{role.label}: calibration produced invalid pump targets")
        return converted, "RPM", f"{assignment.role} pump calibration (g/h → RPM)"

    def compile(self, assignments, configs, monitor_device_ids=()):
        assignments = tuple(assignments)
        roles = [item.role for item in assignments]
        device_ids = [item.device_id for item in assignments]
        if len(roles) != len(set(roles)) or len(device_ids) != len(set(device_ids)):
            raise ValueError("Assign each profile once and use a different device for each profile")
        if any(role not in ROLES for role in roles):
            raise ValueError("Unknown generated profile role")
        missing = self.required_roles() - set(roles)
        if missing:
            raise ValueError("Assign active profiles: " + ", ".join(ROLES[key].label for key in sorted(missing)))
        if not assignments:
            raise ValueError("Assign at least one generated profile")
        for device_id in device_ids:
            if device_id not in configs:
                raise ValueError(f"Assigned device no longer exists: {device_id}")
        for device_id in monitor_device_ids:
            if device_id not in configs or not is_monitor_only(configs[device_id]):
                raise ValueError(f"Monitor must be a read-only device: {device_id}")
        experiments = []
        for name, frame in self.frames.items():
            # Disabled zero-duration generator sections can repeat an identical
            # boundary row. Remove only exact duplicates; conflicting targets
            # at the same instant remain an error.
            frame = frame.drop_duplicates()
            if "t" not in frame or frame.empty:
                raise ValueError(f"{name}: missing generated timeline")
            times = frame["t"].to_numpy(dtype=float) * 3600
            if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0):
                raise ValueError(f"{name}: times must be finite, nonnegative and strictly increasing")
            profiles = []
            for assignment in assignments:
                if assignment.role not in frame:
                    raise ValueError(f"{name}: missing {assignment.role} profile")
                values = frame[assignment.role].to_numpy(dtype=float)
                converted, unit, _ = self._convert(assignment, configs[assignment.device_id], values)
                profile_times = times
                if times[0] > 0:
                    profile_times = np.insert(times, 0, 0.)
                    converted = np.insert(converted, 0, converted[0])
                profiles.append(DeviceProfile(
                    assignment.device_id,
                    tuple(ProfilePoint(float(t), float(v)) for t, v in zip(profile_times, converted)),
                    "step" if assignment.role == "valve_state" else "linear", unit,
                ))
            definition = ExperimentDefinition(name, tuple(profiles), source="doe")
            # The shared validator applies the currently configured device limits.
            ExperimentDefinition.from_frame(definition.to_frame(), configs)
            experiments.append(definition)
        return Campaign("Generated DoE", tuple(experiments), monitor_device_ids=tuple(monitor_device_ids))

    def assignment_details(self, assignments, configs):
        result = []
        for assignment in assignments:
            _, unit, conversion = self._convert(assignment, configs[assignment.device_id], np.array([0.]))
            result.append({"role": assignment.role, "device_id": assignment.device_id,
                           "source_unit": ROLES[assignment.role].unit, "target_unit": unit,
                           "density_g_ml": 1.0 if configs[assignment.device_id]["type"] == "ismatec" else None, "conversion": conversion})
            custom = configs[assignment.device_id].get("pump_calibration")
            if ROLES[assignment.role].unit == "g/h" and custom:
                result[-1].update(calibration_file=custom["source_name"], calibration_sha256=custom["sha256"],
                                  calibration_points=custom["points"], density_g_ml=None)
            elif ROLES[assignment.role].unit == "g/h" and configs[assignment.device_id]["type"] == "masterflex":
                converter_name = {"AS": "as_rate_converter", "solution": "solution_rate_converter",
                                  "solvent": "solvent_rate_converter", "clean_sol": "cleaning_sol_rate_converter"}[assignment.role]
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
                for role, spec in ROLES.items():
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
