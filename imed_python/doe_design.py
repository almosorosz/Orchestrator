"""Build MD, DO and SA trials from the loaded DoE's varying controls."""
from copy import deepcopy

import numpy as np

from bo.decoder import fixed_horizon
from bo.profiles import Profile, make_profile, scalar
from experiment.doe_decisions import LABELS, charge_volume, decision_groups, decision_summary
from .mixing import average_mixing
from .simulated_profile import power_dependent, template_rpm


PROFILE_FIELDS = {"temperature": "TemperatureData", "feed": "FlowRateData"}


def reference_rpm(experiment, process):
    value = scalar(process.get("StirringRPM"))
    if value is None:
        value = scalar(process.get("MixingSummary", {}).get("average_rpm"))
    if value is not None and value >= 0:
        return value
    commanded = experiment.get("CommandedProfiles", {}).get("stirring", {})
    trace = make_profile(commanded.get("points", experiment.get("StirringRateData")), "DoE stirrer",
                         step=commanded.get("step", False))
    if trace is not None and trace.time[-1] > trace.time[0]:
        return trace.total() / (trace.time[-1] - trace.time[0])
    value = template_rpm(process)
    return float(value) if np.isfinite(value) else None


def imed_doe_groups(space, models, experiment, process):
    groups = decision_groups(space, imed=True)
    if "stirring" in groups:
        mixing = process.get("MixingSettings")
        rpm = reference_rpm(experiment, process)
        legacy_power = scalar(process.get("PowerRatio"))
        has_reference = rpm is not None and (mixing is not None and mixing.get("enabled")
                            or mixing is None and rpm > 0 and legacy_power is not None and legacy_power > 0)
        if not has_reference or not any(power_dependent(model) for model in models):
            groups.discard("stirring")
    if "volume" in groups and charge_volume(process) is None:
        groups.discard("volume")
    return groups


def doe_controls(space, groups, *, discrete_seed=False):
    """The profile selectors use existing DoE waveforms, keeping their stage shapes."""
    controls = {}
    for group in LABELS:
        if group not in groups:
            continue
        if group in PROFILE_FIELDS:
            controls[group] = (1., float(len(space["traces"][group])), True)
        else:
            levels = space["levels"][group]
            discrete = group == "seed" and (discrete_seed or levels[0] == 0)
            controls[group] = (1., float(len(levels)), True) if discrete else (levels[0], levels[-1], False)
    return controls


def _linear_points(profile):
    if not profile.step:
        return np.vstack((profile.time, profile.values))
    changes = np.flatnonzero(np.diff(profile.values) != 0) + 1
    before = np.nextafter(profile.time[changes], profile.time[changes - 1])
    times = np.union1d(profile.time, before)
    return np.vstack((times, profile.sample(times)))


class DoeRecipe:
    def __init__(self, experiment, process, space, controls, *, seed_template=None, final_hold_s=0.):
        self.experiment = experiment
        self.process = process
        self.controls = controls
        self.levels = space["levels"]
        self.seed_template = seed_template or {}
        self.rpm = reference_rpm(experiment, process)
        self.volume = charge_volume(process)
        self.traces = {}
        self.baseline = {}
        temperature = make_profile(experiment.get("TemperatureData"), "Experiment template")
        if temperature is None or temperature.time[-1] <= 0:
            raise ValueError("The DoE design needs a positive experiment period.")
        self.horizon = float(temperature.time[-1])
        for key, field in PROFILE_FIELDS.items():
            entries = space["traces"].get(key, [])
            command = experiment.get("CommandedProfiles", {}).get(key, {})
            trace = make_profile(command.get("points", experiment.get(field)), "Experiment template",
                                 step=command.get("step", False))
            if trace is None or len(trace.time) < 2:
                target = trace.values[0] if trace is not None else 0.
                trace = Profile(np.array([0., self.horizon]), np.full(2, target), "Experiment template")
            self.baseline[key] = trace
            if key not in controls:
                # Prefer a common command when the starting DoE held it fixed.
                entries = entries if len(entries) == 1 else []
            profiles = [make_profile(entry["points"], entry["source"], step=entry["step"]) for entry in entries]
            if not profiles:
                profiles = [trace]
            self.traces[key] = profiles
            self.horizon = max(self.horizon, *(float(profile.time[-1]) for profile in profiles))
        # Simulated execution needs a positive constant-temperature, zero-feed tail.
        self.horizon += final_hold_s

    def candidate(self, candidate):
        return self.apply(dict(zip(self.controls, candidate)))

    def apply(self, values):
        proposed, process = dict(self.experiment), dict(self.process)
        for key, field in PROFILE_FIELDS.items():
            if key in values:
                profile = self.traces[key][int(np.rint(values[key])) - 1]
            else:
                profile = self.baseline[key] if key in self.controls else self.traces[key][0]
            points = _linear_points(fixed_horizon(profile, self.horizon, stop=key == "feed"))
            if key == "feed" and points[1, -1] != 0:
                end = points[0, -1]
                points[0, -1] = np.nextafter(end, points[0, -2])
                if points[0, -1] <= points[0, -2]:
                    points = points[:, :-1]
                points = np.column_stack((points, [end, 0.]))
            proposed[field] = points
        if "seed" in values:
            selected = values["seed"]
            if self.controls["seed"][2]:
                selected = self.levels["seed"][int(np.rint(selected)) - 1]
            process["SeedMass"] = float(selected)
            if selected > 0 and not np.size(proposed.get("InitialCSDsData", [])):
                if not np.size(self.seed_template.get("InitialCSDsData", [])):
                    raise ValueError("A positive DoE seed level requires a seed size distribution.")
                proposed.update(self.seed_template)
        if "volume" in values:
            volume = float(values["volume"])
            factor = volume / self.volume
            process["SolvMass"] *= factor
            process["AntiSolvMass"] *= factor
            process.update(InitialVolume_ml=volume, RequestedInitialVolume_ml=volume, VolumeFromDesignConditions=True)
        rpm = float(values["stirring"]) if "stirring" in values else self.rpm
        if "stirring" in values:
            process["StirringRPM"] = rpm
            proposed["ProposedStirringRPM"] = rpm
        proposed.pop("AddedAntisolventData", None)
        mixing = process.get("MixingSettings")
        if mixing is not None:
            mixing = deepcopy(mixing)
            if "volume" in values and mixing["volume_mode"] == "fixed":
                mixing["volume_ml"] *= process["InitialVolume_ml"] / self.volume
            process = average_mixing(proposed, process, mixing, mean_rpm=rpm, source="DoE design")
        elif "PowerRatio" in process:
            if "stirring" in values:
                process["PowerRatio"] *= (rpm / self.rpm)**3
            if "volume" in values:
                process["PowerRatio"] *= self.volume / process["InitialVolume_ml"]
        if "InitialConcentration" not in proposed:
            concentration = np.asarray(proposed.get("ConcentrationData", [])).reshape(2, -1)
            if concentration.size:
                proposed["InitialConcentration"] = float(concentration[1, 0])
        return proposed, process


def seed_template(case):
    return next(({key: experiment[key] for key in ("InitialCSDsData", "InitialD10D50D90") if key in experiment}
                 for experiment, process in zip(case["ExpData"], case["ProcInf"])
                 if process.get("SeedMass", 0) > 0 and np.size(experiment.get("InitialCSDsData", []))), {})


def imed_recipe_profiles(experiment, source):
    profiles = {key: make_profile(experiment.get(field), source) for key, field in PROFILE_FIELDS.items()}
    if not all(profiles.values()):
        raise ValueError("The IMED recipe needs finite temperature and antisolvent profiles.")
    if "ProposedStirringRPM" in experiment:
        horizon = profiles["temperature"].time[-1]
        rpm = float(experiment["ProposedStirringRPM"])
        profiles["stirring"] = Profile(np.array([0., horizon]), np.full(2, rpm), source)
    return profiles


def design_note(space, controls):
    return f"DoE decision variables: {decision_summary(space, controls)}. Profile choices use the waveforms in the DoE."
