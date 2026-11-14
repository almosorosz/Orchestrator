"""Load, encode and assess profiles in BO's canonical units."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from imed_python.campaign_import import read_json
from imed_python.session import display_name
from imed_python.simulated_profile import STIRRING_FIELDS

CHANNELS = {
    "temperature": ("Temperature", "°C", .15),
    "feed": ("Antisolvent", "g/s", .001),
    "seed": ("Seed mass", "g", 0.),
    "stirring": ("Stirring", "RPM", 1.),
}
CONSTRAINTS = {
    "duration": ("Batch duration", "s"),
    "cooling": ("Cooling rate", "°C/min"),
    "seed": ("Seed mass", "g"),
    "feed": ("Antisolvent rate", "g/s"),
}
TARGETS = {
    "yield": ("Yield", "%", ("max",)),
    "duration": ("Batch duration", "s", ("min",)),
    **{key: (key.upper(), "µm", ("min", "max", "match")) for key in ("d10", "d50", "d90")},
}


def scalar(value):
    try:
        data = np.asarray(value, dtype=float)
        if data.size == 1 and np.isfinite(data).all():
            return float(data.item())
    except (ValueError, TypeError):
        pass
    return None


def pairs(value):
    try:
        data = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return None
    if not data.size:
        return None
    if data.ndim == 1 and data.size == 2:
        data = data[:, None]
    if (data.ndim != 2 or data.shape[0] != 2 or not np.isfinite(data).all()
            or data[0, 0] < 0 or np.any(np.diff(data[0]) <= 0)):
        return None
    return data


@dataclass(frozen=True)
class Profile:
    time: np.ndarray
    values: np.ndarray
    source: str
    step: bool = False

    def sample(self, time):
        if self.step:
            index = np.clip(np.searchsorted(self.time, time, side="right") - 1, 0, len(self.time) - 1)
            return self.values[index]
        return np.interp(time, self.time, self.values)

    def total(self):
        rates = self.values[:-1] if self.step else (self.values[:-1] + self.values[1:]) / 2
        return float(np.dot(np.diff(self.time), rates))

    def window(self, start, end):
        time = np.r_[start, self.time[(self.time > start) & (self.time < end)], end]
        return Profile(time - start, self.sample(time), self.source, self.step)

    def payload(self):
        return {"points": [self.time.tolist(), self.values.tolist()], "source": self.source, "step": self.step}


def make_profile(points, source, *, step=False):
    trace = pairs(points)
    return Profile(*trace, source, step) if trace is not None else None


def campaign_profiles(experiment, process, definition):
    recording = experiment.get("RecordingFile")
    if not isinstance(recording, str):
        return {}
    metadata = read_json(Path(recording).with_name("run.json"))
    experiment_id = metadata.get("experiment_id")
    spec = next((entry for entry in definition.get("campaign", {}).get("experiments", [])
                 if entry.get("experiment_id") == experiment_id), {})
    demo = spec.get("imed_simulation", {})
    devices = definition.get("devices", {})
    controls = demo.get("controls", {})
    recorded_temp = pairs(experiment.get("TemperatureData"))
    duration = float(recorded_temp[0, -1]) if recorded_temp is not None else 0.
    start = scalar(experiment.get("RecordingStart_s")) or 0.
    def unique_device(types, *, antisolvent=False):
        candidates = [p["device_id"] for p in spec.get("profiles", [])
                      if devices.get(p["device_id"], {}).get("type") in types
                      and (not antisolvent or "AS" in p["device_id"].upper())]
        return candidates[0] if len(candidates) == 1 else None

    associations = {
        "temperature": controls.get("T") or unique_device(("huber",)),
        "feed": controls.get("AS") or unique_device(("ismatec", "masterflex"), antisolvent=True),
        "stirring": unique_device(("jkem",)),
    }
    found = {}
    for entry in spec.get("profiles", []):
        name = next((key for key, device in associations.items() if device == entry["device_id"]), None)
        if name is None or duration <= 0:
            continue
        unit = entry.get("unit", "").strip().lower().replace("°", "")
        if (name == "temperature" and unit != "c") or (name == "stirring" and unit != "rpm"):
            continue
        scale = 1.
        if name == "feed":
            density = scalar(process.get("AntiSolventDensity"))
            factors = {"g/s": 1., "g/h": 1 / 3600, "g/min": 1 / 60}
            if density is not None and density > 0:
                factors["ml/min"] = density / 60
            if unit not in factors:
                continue
            scale = factors[unit]
        profile = make_profile([[p["time_s"] for p in entry["points"]],
                                [p["target"] * scale for p in entry["points"]]],
                               f"Command · {entry['device_id']}", step=entry.get("interpolation") == "step")
        end = start + duration
        if profile and profile.time[0] <= start + 1e-6 and profile.time[-1] >= end - 1e-6:
            profile = profile.window(start, end)
            if name == "temperature" or np.all(profile.values >= 0):
                found[name] = profile
    return found


def profile_family(profile, channel):
    """Compare stage structure before aligning variables across experiments."""
    if channel == "seed":
        return ("scalar",)
    change = np.sign(np.diff(profile.values)).astype(int)
    return ("step" if profile.step else "linear", *change.tolist())


def case_runs(case, folder=None):
    definition = read_json(Path(folder) / "definition.json") if folder else {}
    process_records = case.get("ProcInf", [])
    runs = []
    for index, exp in enumerate(case.get("ExpData", [])):
        if index < len(process_records):
            process = process_records[index]
        else:
            process = {}
        temperature = make_profile(exp.get("TemperatureData"), "Recorded or imported temperature")
        if temperature is None or len(temperature.time) < 2 or temperature.time[-1] <= 0:
            continue
        end = float(temperature.time[-1])
        profiles = {"temperature": temperature.window(0, end)}
        for key, field in (("feed", "FlowRateData"), ("stirring", "StirringRateData")):
            profile = make_profile(exp.get(field), "Recorded or imported profile")
            if profile and profile.time[0] == 0 and (len(profile.time) == 1 or profile.time[-1] >= end):
                profiles[key] = profile.window(0, end)
        seed = scalar(process.get("SeedMass"))
        if seed is not None and seed >= 0:
            profiles["seed"] = Profile(np.array([0.]), np.array([seed]), "Process setting")
        if "stirring" not in profiles:
            rpm = next((value for key in STIRRING_FIELDS
                        if (value := scalar(process.get(key))) is not None and value >= 0), None)
            if rpm is not None:
                profiles["stirring"] = Profile(np.array([0., end]), np.array([rpm, rpm]), "Process setting")
        if folder:
            profiles.update(campaign_profiles(exp, process, definition))
        conc = pairs(exp.get("ConcentrationData"))
        initial = scalar(exp.get("InitialConcentration"))
        if initial is None and conc is not None and conc[0, 0] == 0:
            initial = float(conc[1, 0])
        final = None
        if conc is not None and len(conc[0]) > 1 and np.isclose(conc[0, -1], end, atol=1e-6, rtol=0):
            final = float(conc[1, -1])
        sizes = np.asarray(exp.get("FinalD10D50D90", []), dtype=float).ravel()
        metrics = {"duration": end, "yield_actual": None, "yield_theoretical": None,
                   **dict.fromkeys(("d10", "d50", "d90"))}
        if sizes.size == 3 and np.isfinite(sizes).all() and 0 < sizes[0] <= sizes[1] <= sizes[2]:
            metrics.update(zip(("d10", "d50", "d90"), map(float, sizes)))
        # Use the measured feed for mass balance when available, independently of the command used for encoding.
        measured_feed = make_profile(exp.get("FlowRateData"), "Recorded or imported profile")
        feed = measured_feed.window(0, end) if measured_feed and measured_feed.time[0] == 0 and (
            len(measured_feed.time) == 1 or measured_feed.time[-1] >= end) else profiles.get("feed")
        metrics.update(yields(case.get("Solubility"), process, initial, final, temperature.values[-1], feed))
        runs.append({"name": display_name(exp, f"Experiment {index + 1}"), "metrics": metrics,
                     "profiles": {channel: profile.payload() for channel, profile in profiles.items()}})
    if not runs:
        raise ValueError("No experiments with a usable temperature timeline were found.")
    return runs


def yields(solubility, process, initial, final, temperature, feed):
    solvent = scalar(process.get("SolvMass"))
    anti = scalar(process.get("AntiSolvMass"))
    if (solvent is None or solvent <= 0 or anti is None or anti < 0 or initial is None
            or not 0 < initial < 1 or feed is None or np.any(feed.values < 0)):
        return {}
    carrier = solvent + anti + feed.total()
    initial_api = (solvent + anti) * initial / (1 - initial)
    results = {}
    if final is not None and 0 <= final < 1:
        results["yield_actual"] = 100 * (1 - carrier * final / (1 - final) / initial_api)
    if isinstance(solubility, dict):
        coeff = [scalar(solubility.get(key)) for key in "abcdefgh"]
        if None not in coeff and temperature > -273.15:
            t = temperature + 273.15
            x = 1 - solvent / carrier
            basis = [1, 1 / t, np.log(t), x, np.exp(x), x*x, x / t, np.exp(x) / t]
            log_fraction = float(np.dot(coeff, basis))
            if log_fraction < 0:
                sat = float(np.exp(log_fraction))
                results["yield_theoretical"] = max(0., 100 * (1 - carrier * sat / (1 - sat) / initial_api))
    return results


def encode(profile, tolerance, channel):
    """Keep endpoints and split at the largest value error until the tolerance is met."""
    t, y = profile.time, profile.values
    if channel == "seed" or len(t) < 3:
        return profile
    if profile.step:
        keep = np.r_[True, np.diff(y) != 0]
        keep[-1] = True
        return replace(profile, time=t[keep], values=y[keep])
    keep = {0, len(t) - 1}
    pending = [(0, len(t) - 1)]
    while pending:
        first, last = pending.pop()
        if last - first <= 1:
            continue
        slope = (y[last] - y[first]) / (t[last] - t[first])
        errors = np.abs(y[first+1:last] - (y[first] + slope * (t[first+1:last] - t[first])))
        split = first + 1 + int(errors.argmax())
        if errors.max() > tolerance:
            keep.add(split)
            pending.extend(((first, split), (split, last)))
    indices = sorted(keep)
    candidate = replace(profile, time=t[indices], values=y[indices])
    # Retain the original knots if a feed approximation would change the integrated dose.
    if channel == "feed" and not np.isclose(candidate.total(), profile.total(), rtol=1e-6, atol=1e-8):
        return profile
    return candidate


def components(channel, profile):
    if channel == "seed":
        return {"Mass": float(profile.values[0])}
    values = {"Start value": float(profile.values[0])}
    for index, (duration, value) in enumerate(zip(np.diff(profile.time), profile.values[1:]), 1):
        values[f"Segment {index} duration"] = float(duration)
        values[f"Segment {index} end value"] = float(value)
    return values


def constraint_values(channel, profile):
    if channel == "temperature" and len(profile.time) > 1:
        rates = -60 * np.diff(profile.values) / np.diff(profile.time)
        return "cooling", rates[rates > 0]
    if channel == "feed":
        return "feed", profile.values[profile.values > 0]
    if channel == "seed":
        return "seed", profile.values
    return None, ()


def violations(values, lower, upper):
    return any((lower is not None and value < lower) or (upper is not None and value > upper)
               for value in values)
