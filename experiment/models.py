"""Shared, GUI-independent experiment definitions and execution records."""
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import math

import pandas as pd
from hardware_devices.limits import DEFAULT_LIMITS, target_limits, target_allowed
from hardware_devices.capabilities import is_monitor_only

DEVICE_LIMITS = DEFAULT_LIMITS


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class ProfilePoint:
    time_s: float
    target: float


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    points: tuple[ProfilePoint, ...]
    interpolation: str = "linear"
    unit: str = ""

    def __post_init__(self):
        if not self.device_id or not self.points:
            raise ValueError("A profile needs a device ID and at least one point")
        if self.interpolation not in {"linear", "step"}:
            raise ValueError("Interpolation must be linear or step")
        previous = -1.0
        for point in self.points:
            if not math.isfinite(point.time_s) or point.time_s < 0 or point.time_s <= previous:
                raise ValueError("Profile times must be finite, nonnegative and strictly increasing")
            if not math.isfinite(point.target):
                raise ValueError("Profile targets must be finite")
            previous = point.time_s


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    profiles: tuple[DeviceProfile, ...]
    source: str = "manual"
    experiment_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        ids = [profile.device_id for profile in self.profiles]
        if not self.name or not ids or len(ids) != len(set(ids)):
            raise ValueError("An experiment needs a name and unique device profiles")

    @classmethod
    def from_frame(cls, frame, device_configs, name="Experiment", source="manual"):
        if "time_s" not in frame or frame.empty or not frame.columns.is_unique:
            raise ValueError("Experiment requires unique columns and a nonempty time_s timeline")
        profiles = []
        for device_id in frame.columns:
            if device_id == "time_s":
                continue
            if device_id not in device_configs:
                raise ValueError(f"Unknown device: {device_id}")
            config = device_configs[device_id]
            if config["type"] not in DEVICE_LIMITS or is_monitor_only(config):
                raise ValueError(f"Read-only device cannot receive a profile: {device_id}")
            points = tuple(ProfilePoint(float(t), float(v)) for t, v in zip(frame["time_s"], frame[device_id]))
            limits = target_limits(config)
            minimum, maximum, unit = limits["min"], limits["max"], limits["unit"]
            if any(not target_allowed(p.target, limits) for p in points):
                raise ValueError(f"{device_id}: targets must be {minimum:g}–{maximum:g} {unit}; "
                                 f"minimum nonzero magnitude {limits['min_nonzero']:g}. Edit Device Limits to override.")
            if config["type"] == "nidaq_valve" and any(p.target not in (0, 1) for p in points):
                raise ValueError(f"Valve {device_id} requires binary targets")
            profiles.append(DeviceProfile(device_id, points, "step" if config["type"] == "nidaq_valve" else "linear", unit))
        return cls(name=name, profiles=tuple(profiles), source=source)

    def to_frame(self):
        import numpy as np
        times = sorted({p.time_s for profile in self.profiles for p in profile.points})
        result = {"time_s": times}
        for profile in self.profiles:
            x = [p.time_s for p in profile.points]
            y = [p.target for p in profile.points]
            if profile.interpolation == "step":
                idx = np.clip(np.searchsorted(x, times, side="right") - 1, 0, len(y) - 1)
                result[profile.device_id] = np.asarray(y)[idx]
            else:
                result[profile.device_id] = np.interp(times, x, y)
        return pd.DataFrame(result)


@dataclass(frozen=True)
class Campaign:
    name: str
    experiments: tuple[ExperimentDefinition, ...]
    campaign_id: str = field(default_factory=lambda: uuid4().hex)
    monitor_device_ids: tuple[str, ...] = ()


@dataclass
class ExperimentRun:
    experiment_id: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    status: RunStatus = RunStatus.PENDING
    error: str = ""
