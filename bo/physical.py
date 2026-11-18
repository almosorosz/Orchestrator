"""Ordered physical profile stages for BO and its profile editor."""

from dataclasses import dataclass, replace
from math import fsum, isfinite
from numbers import Real

import numpy as np

from .profiles import CHANNELS, Profile


@dataclass(frozen=True)
class Stage:
    duration_s: float
    start: float
    end: float


@dataclass(frozen=True)
class PhysicalControl:
    key: str
    label: str
    unit: str
    kind: str
    stage: int | None
    channel: str = ""


def validate_stages(channel, stages, horizon=None):
    """Validate units, continuity, and the fixed horizon for physical stages."""
    if channel not in ("temperature", "feed"):
        raise ValueError("Physical stages apply to temperature or antisolvent.")
    stages = tuple(stages)
    if not stages:
        raise ValueError("Add at least one stage to the profile.")
    for index, stage in enumerate(stages, 1):
        if not isinstance(stage, Stage):
            raise ValueError(f"Stage {index} needs a duration and start and end values.")
        numbers = (stage.duration_s, stage.start, stage.end)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
               for value in numbers):
            raise ValueError(f"Stage {index} needs finite numeric values.")
        if stage.duration_s <= 0:
            raise ValueError(f"Stage {index} duration must be positive.")
        if channel == "temperature" and min(stage.start, stage.end) <= -273.15:
            raise ValueError(f"Stage {index} temperature must exceed absolute zero.")
        if channel == "feed" and min(stage.start, stage.end) < 0:
            raise ValueError(f"Stage {index} antisolvent rate cannot be negative.")
    if channel == "temperature" and any(a.end != b.start for a, b in zip(stages, stages[1:])):
        raise ValueError("Each temperature stage must start at the previous stage’s end temperature.")
    if horizon is not None:
        if (isinstance(horizon, bool) or not isinstance(horizon, Real)
                or not isfinite(horizon) or horizon <= 0):
            raise ValueError("The fixed batch horizon must be a positive number of seconds.")
        try:
            total = fsum(stage.duration_s for stage in stages)
        except OverflowError as error:
            raise ValueError("Stage durations exceed the fixed batch horizon.") from error
        if total - horizon > np.spacing(horizon) * max(8, len(stages)):
            raise ValueError(f"Stage durations total {total:g} s and exceed the fixed batch horizon of {horizon:g} s.")
    return stages


def profile_stages(profile, count=8, *, channel=None):
    """Encode a trace's breakpoints, or simplify to a bounded starting template."""
    if count is not None and (not isinstance(count, int) or count < 1):
        raise ValueError("The number of stages must be a positive integer.")
    time = np.asarray(profile.time, dtype=float)
    values = np.asarray(profile.values, dtype=float)
    if (time.ndim != 1 or time.shape != values.shape or len(time) < 2 or time[0] != 0
            or not np.isfinite(time).all() or not np.isfinite(values).all() or np.any(np.diff(time) <= 0)):
        raise ValueError("Use a profile with finite values and increasing times starting at 0 s.")

    if profile.step and channel != "temperature":
        # Merge adjacent equal-rate command intervals before simplifying.
        edges = np.r_[0, np.flatnonzero(np.diff(values[:-1]) != 0) + 1, len(time) - 1]
        time, values = time[edges], values[edges]
        if count is None or len(time) <= count + 1:
            return tuple(Stage(float(dt), float(value), float(value))
                         for dt, value in zip(np.diff(time), values))

    if count is None:
        if profile.step and channel == "temperature":
            changes = np.flatnonzero(np.diff(values) != 0) + 1
            ts, ys = [0.], [values[0]]
            for index in changes:
                before = np.nextafter(time[index], ts[-1])
                if before > ts[-1]:
                    ts.append(before)
                    ys.append(values[index - 1])
                ts.append(time[index])
                ys.append(values[index])
            if ts[-1] < time[-1]:
                ts.append(time[-1])
                ys.append(values[-1])
            time, values = np.asarray(ts), np.asarray(ys)
        return _stages_at_breakpoints(time, values, feed=channel == "feed")

    # Add knots where the current linear approximation has the largest residual.
    kept = [0, len(time) - 1]
    value_epsilon = 1e-12 * max(1., float(np.max(np.abs(values))))
    while len(kept) < count + 1:
        error = np.abs(np.interp(time, time[kept], values[kept]) - values)
        index = int(np.argmax(error))
        if error[index] <= value_epsilon:
            break
        kept.append(index)
        kept.sort()
    return tuple(Stage(float(time[b] - time[a]), float(values[a]), float(values[b]))
                 for a, b in zip(kept, kept[1:]))


def _stages_at_breakpoints(time, values, *, feed):
    intervals = np.diff(time)
    fractions = intervals[:-1] / (intervals[:-1] + intervals[1:])
    interpolated = values[:-2] + fractions * (values[2:] - values[:-2])
    epsilon = 1e-12 * max(1., float(np.max(np.abs(values))))
    bends = np.flatnonzero(np.abs(values[1:-1] - interpolated) > epsilon) + 1
    jumps = np.flatnonzero((np.nextafter(time[:-1], time[1:]) == time[1:]) & (values[:-1] != values[1:])) if feed else []
    edges = np.unique(np.r_[0, bends, jumps, np.asarray(jumps) + 1, len(time) - 1]).astype(int)
    jumps = set(jumps)
    stages = []
    for first, last in zip(edges, edges[1:]):
        duration = float(time[last] - time[first])
        if stages and first in jumps and last == first + 1:
            # Feed stages can jump at a boundary, so a one-ULP change needs no artificial ramp.
            previous = stages[-1]
            stages[-1] = replace(previous, duration_s=previous.duration_s + duration)
        else:
            stages.append(Stage(duration, float(values[first]), float(values[last])))
    return tuple(stages)


def physical_profile(channel, stages, horizon):
    """Decode an edited profile on a fixed clock with a final hold or feed stop."""
    stages = validate_stages(channel, stages, horizon)
    boundaries = np.r_[0., np.cumsum([stage.duration_s for stage in stages])]
    boundaries[-1] = min(fsum(stage.duration_s for stage in stages), horizon)
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("A stage duration is too short to resolve on this time axis.")
    times, values = [0.], [stages[0].start]
    step = channel == "feed" and all(stage.start == stage.end for stage in stages)
    for index, stage in enumerate(stages):
        end_time = float(boundaries[index + 1])
        next_start = stages[index + 1].start if index + 1 < len(stages) else stage.end
        if channel == "feed" and index + 1 == len(stages):
            next_start = 0.
        if next_start != stage.end and not step:
            # Keep both sides of a discontinuity on a strictly increasing time axis.
            before = np.nextafter(end_time, times[-1])
            if before == times[-1]:
                raise ValueError(f"Stage {index + 1} duration is too short to resolve its feed change.")
            times.append(float(before))
            values.append(stage.end)
        times.append(end_time)
        values.append(next_start)
    if times[-1] < horizon:
        times.append(horizon)
        values.append(values[-1])
    source = f"{CHANNELS[channel][0]} stages"
    return Profile(np.asarray(times), np.asarray(values), source, step=step)


def physical_controls(channel, stages):
    """Describe editable physical variables in stage order."""
    stages = validate_stages(channel, stages)
    controls = []
    _, unit, _ = CHANNELS[channel]
    for index, stage in enumerate(stages):
        label = f"Stage {index + 1}"
        key = f"stage:{index}"
        controls.append(PhysicalControl(f"{key}:duration", f"{label} duration", "s", "duration", index))
        if channel == "feed" and stage.start == stage.end:
            controls.append(PhysicalControl(f"{key}:start", f"{label} feed rate", unit, "start", index))
            continue
        if index == 0 or channel == "feed":
            controls.append(PhysicalControl(f"{key}:start", f"{label} start", unit, "start", index))
        controls.append(PhysicalControl(f"{key}:end", f"{label} end", unit, "end", index))
    return tuple(controls)


def grouped_controls(channel, stages):
    """Describe whole-profile controls supported by the supplied stage shapes."""
    stages = validate_stages(channel, stages)
    _, unit, _ = CHANNELS[channel]
    if channel == "temperature":
        final = PhysicalControl("final_temperature", "Final temperature", unit, "final", None, channel)
        if _initial_holds(channel, stages) == len(stages):
            return (final,)
        return (PhysicalControl("initial_delay", "Cooling start delay", "s", "delay", None, channel), final)
    if not any(_feed_on(stage) for stage in stages):
        return ()
    return (PhysicalControl("initial_delay", "Antisolvent start delay", "s", "delay", None, channel),
            PhysicalControl("on_time", "Antisolvent on time", "s", "on_time", None, channel),
            PhysicalControl("total_dose", "Total antisolvent dose", "g", "dose", None, channel))


def _feed_on(stage):
    return stage.start > 0 or stage.end > 0


def _initial_holds(channel, stages):
    for index, stage in enumerate(stages):
        active = (stage.start != stage.end) if channel == "temperature" else _feed_on(stage)
        if active:
            return index
    return len(stages)


def _dose(stages):
    return fsum(stage.duration_s * (stage.start * .5 + stage.end * .5) for stage in stages)


def _group_value(stages, control):
    kind = control.kind
    if kind == "delay":
        return fsum(stage.duration_s for stage in stages[:_initial_holds(control.channel, stages)])
    if kind == "final":
        return stages[-1].end
    if kind == "on_time":
        return fsum(stage.duration_s for stage in stages if _feed_on(stage))
    return _dose(stages)


def control_values(stages, controls):
    """Return the controls’ values in the same order as the controls."""
    return np.array([_group_value(stages, control) if control.stage is None
                     else getattr(stages[control.stage], "duration_s" if control.kind == "duration" else control.kind)
                     for control in controls], dtype=float)


def _set_final(stages, value):
    last_ramp = next((index for index in range(len(stages) - 1, -1, -1)
                      if stages[index].start != stages[index].end), None)
    if last_ramp is None:
        return [replace(stage, start=value, end=value) for stage in stages]
    return [*stages[:last_ramp], replace(stages[last_ramp], end=value),
            *(replace(stage, start=value, end=value) for stage in stages[last_ramp + 1:])]


def _set_delay(channel, stages, value):
    count = _initial_holds(channel, stages)
    if count == len(stages):
        if value == 0:
            return stages
        raise ValueError("A start delay requires an active stage.")
    if value == 0:
        return stages[count:]
    if count == 0:
        initial = stages[0].start if channel == "temperature" else 0.
        return [Stage(value, initial, initial), *stages]
    total = fsum(stage.duration_s for stage in stages[:count])
    return [*(replace(stage, duration_s=value * stage.duration_s / total) for stage in stages[:count]),
            *stages[count:]]


def _set_on_time(stages, value):
    if value == 0:
        return [replace(stage, start=0., end=0.) for stage in stages]
    total = fsum(stage.duration_s for stage in stages if _feed_on(stage))
    if total <= 0:
        raise ValueError("Antisolvent on time requires at least one active feed stage.")
    return [replace(stage, duration_s=stage.duration_s * value / total) if _feed_on(stage) else stage
            for stage in stages]


def _set_dose(stages, value):
    total = _dose(stages)
    if total <= 0:
        if value > 0:
            raise ValueError("A positive antisolvent dose requires at least one active feed stage.")
        return stages
    return [replace(stage, start=stage.start * value / total, end=stage.end * value / total)
            for stage in stages]


def apply_controls(channel, stages, controls, values):
    """Apply control values to a template and reconnect temperature stages."""
    stages = list(validate_stages(channel, stages))
    controls, values = tuple(controls), tuple(values)
    if len(controls) != len(values):
        raise ValueError("Provide one value for each selected physical control.")
    holds = {i for i, stage in enumerate(stages) if stage.start == stage.end}
    explicit_ends, grouped = set(), {}
    for control, value in zip(controls, values):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{control.label} must be finite.")
        if control.stage is None:
            if control.kind != "final" and value < 0:
                raise ValueError(f"{control.label} cannot be negative.")
            grouped[control.kind] = value
            continue
        index = control.stage
        field = "duration_s" if control.kind == "duration" else control.kind
        if control.kind == "end" and value != stages[index].end:
            explicit_ends.add(index)
        stage = replace(stages[index], **{field: value})
        if channel == "feed" and control.kind == "start" and index in holds:
            stage = replace(stage, end=value)
        stages[index] = stage
    if channel == "temperature":
        for index, stage in enumerate(stages):
            start = stages[index - 1].end if index else stage.start
            end = start if index in holds and index not in explicit_ends else stage.end
            stages[index] = replace(stage, start=start, end=end)
        if "final" in grouped:
            stages = _set_final(stages, grouped["final"])
    if "delay" in grouped:
        stages = _set_delay(channel, stages, grouped["delay"])
    if "on_time" in grouped:
        stages = _set_on_time(stages, grouped["on_time"])
    if "dose" in grouped:
        if any(control.stage is not None and control.kind in ("start", "end") for control in controls):
            raise ValueError("Vary total antisolvent dose or absolute stage rates one at a time.")
        if grouped["dose"] == 0 and grouped.get("on_time", 0.) > 0:
            raise ValueError("A zero dose cannot have a positive antisolvent on time.")
        stages = _set_dose(stages, grouped["dose"])
    return validate_stages(channel, stages)
