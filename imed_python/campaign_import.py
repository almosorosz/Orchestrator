"""Import measured experiment data from a CrystalLoop recorder campaign."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


CHANNELS = ("Temperature", "Concentration", "FBRM", "D10", "D50", "D90", "AS scale")
ALIASES = {
    "Temperature": ("T1.temp_external", "T1.temp_internal", "TEM_MAIN.temp_external",
                    "TEM_MAIN.temp_internal", "T", "temperature_main"),
    "Concentration": ("IR1.value", "Concentration", "c"),
    "FBRM": ("FBRM1.value", "FBRM"),
    "D10": ("D101.value", "D10"), "D50": ("D501.value", "D50"), "D90": ("D901.value", "D90"),
    "AS scale": ("WAS1.weight_g", "SCL_AS.weight_g", "Scale"),
}
EMPTY = np.empty((2, 0))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def campaign_runs(folder):
    folder = Path(folder)
    definition_path = folder / "definition.json"
    if not definition_path.is_file():
        raise ValueError("Select a CrystalLoop campaign folder containing definition.json.")
    definition = read_json(definition_path)
    names = {item["experiment_id"]: item.get("name") for item in
             definition.get("campaign", {}).get("experiments", []) if "experiment_id" in item}
    runs = []
    for path in sorted(folder.glob("*/run.json")):
        metadata = read_json(path)
        if metadata.get("status") != "completed" or not metadata.get("recording_complete"):
            continue
        file = path.with_name("execution.csv")
        if not file.is_file():
            raise ValueError(f"Completed run {path.parent.name} is missing execution.csv.")
        name = (metadata.get("batch_name") or names.get(metadata.get("experiment_id"))
                or path.parent.name)
        runs.append((metadata.get("experiment_number"), name, file))
    if not runs:
        raise ValueError("The campaign has no completed runs with finished recordings.")
    return sorted(runs, key=lambda run: (run[0] is None, run[0] or 0, run[1]))


def channel_options(runs):
    """List actual numeric channels and identify conventional CrystalLoop IDs."""
    columns = set()
    for _, _, file in runs:
        with file.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                for key, value in row.items():
                    if key and key not in columns and key not in ("elapsed_s", "utc"):
                        try:
                            if np.isfinite(float(value)):
                                columns.add(key)
                        except (ValueError, TypeError):
                            pass
    defaults = {role: next((name for name in ALIASES[role] if name in columns), "")
                for role in CHANNELS}
    return sorted(columns), defaults


def scale_to_feed(scale, *, direction, smoothing):
    """Estimate a nonnegative linear feed profile whose integral matches net scale change."""
    time, mass = scale
    radius = smoothing // 2
    if radius:
        padded = np.pad(mass, radius, mode="edge")
        mass = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1), axis=1)
    change = direction * (mass - mass[0])
    total = change[-1]
    if total < -1e-9:
        raise ValueError("The AS scale moved opposite the selected direction. Check the run window and scale.")
    added = np.maximum.accumulate(np.clip(change, 0, max(total, 0)))
    midpoint = (time[:-1] + time[1:]) / 2
    knots = np.sort(np.r_[time, midpoint])
    rate = np.maximum(PchipInterpolator(time, added).derivative()(knots), 0)
    integral = np.trapezoid(rate, knots)
    if total > 0 and integral > 0:
        rate *= total / integral
    return np.vstack((time, added)), np.vstack((knots, rate))


def seed_distribution(percentiles):
    d10, d50, d90 = percentiles
    if not 0 < d10 < d50 < d90:
        raise ValueError("Enter positive, increasing seed D10, D50 and D90.")
    anchors = np.array([0.5 * d10, d10, d50, d90, 2 * d90])
    cdf = PchipInterpolator(np.log(anchors), [0, 0.1, 0.5, 0.9, 1])
    size = np.unique(np.r_[np.linspace(anchors[0], anchors[-1], 1001), anchors])
    density = np.maximum(cdf.derivative()(np.log(size)) / size, 0)
    density[[0, -1]] = 0
    return np.vstack((size, density))


def import_run(file, mapping, *, start, end, concentration_multiplier, scale_direction,
               smoothing, initial_concentration, seed_percentiles, no_feed,
               feed_channel=None, feed_multiplier=1.):
    if feed_channel:
        mapping = {**mapping, "AS flow": feed_channel}
    selected = {role: {} for role in (*CHANNELS, "AS flow")}
    with file.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "elapsed_s" not in reader.fieldnames:
            raise ValueError("execution.csv is missing elapsed_s.")
        for row in reader:
            try:
                time = float(row["elapsed_s"])
            except (ValueError, TypeError):
                continue
            if not np.isfinite(time) or time < start or (end is not None and time > end):
                continue
            for role, channel in mapping.items():
                try:
                    value = float(row.get(channel, ""))
                except (ValueError, TypeError):
                    continue
                if not np.isfinite(value):
                    continue
                # Ignore telemetry from an explicitly disconnected device.
                if "." in channel and str(row.get(channel.rsplit(".", 1)[0] + ".connected", "")).lower() in ("false", "0"):
                    continue
                selected[role][time] = value
    origin = min(selected["Temperature"], default=start)

    def series(role):
        pairs = sorted((t - origin, value) for t, value in selected[role].items() if t >= origin)
        return np.asarray(pairs, dtype=float).T if pairs else EMPTY.copy()

    temperature = series("Temperature")
    if temperature.shape[1] < 2 or np.any(temperature[1] <= -273.15):
        raise ValueError("Two or more valid measured temperature samples are required.")
    duration = temperature[0, -1]

    def within_window(role):
        points = series(role)
        return points[:, points[0] <= duration]

    concentration = within_window("Concentration")
    concentration[1] *= concentration_multiplier
    if np.any((concentration[1] < 0) | (concentration[1] >= 1)):
        raise ValueError("Converted concentration must be in [0, 1) g/g solution. Check its units.")
    if initial_concentration is None:
        if not concentration.size or concentration[0, 0] != 0:
            raise ValueError("Enter an initial concentration when the channel has no sample at the run start.")
        initial_concentration = float(concentration[1, 0])
    scale = within_window("AS scale")
    if no_feed:
        added = feed = np.array([[0., duration], [0., 0.]])
    elif feed_channel:
        feed = within_window("AS flow")
        feed[1] *= feed_multiplier
        if (feed.shape[1] < 2 or feed[0, 0] != 0 or feed[0, -1] != duration
                or np.any(feed[1] < 0) or scale.shape[1] < 2):
            raise ValueError("Assigned AS flow and reservoir scale must cover the crystallization window.")
        added = np.vstack((scale[0], scale_direction * (scale[1] - scale[1, 0])))
    else:
        if scale.shape[1] < 2:
            raise ValueError("Map an AS scale with two or more readings, or select no AS feed.")
        added, feed = scale_to_feed(scale, direction=scale_direction, smoothing=smoothing)
        if feed[0, 0] > 0:
            feed = np.column_stack(([0., 0.], [np.nextafter(feed[0, 0], 0), 0.], feed))
        if feed[0, -1] < duration:
            feed = np.column_stack((feed, [np.nextafter(feed[0, -1], duration), 0.], [duration, 0.]))
            # A boundary one representable step away does not need a duplicate endpoint.
            feed = feed[:, np.r_[True, np.diff(feed[0]) > 0]]

    sizes = {role: within_window(role) for role in ("D10", "D50", "D90")}
    lookup = [{t: y for t, y in points.T} for points in sizes.values()]
    final = []
    for t in sorted(set(lookup[0]).intersection(*lookup[1:]), reverse=True):
        values = [channel[t] for channel in lookup]
        if 0 < values[0] <= values[1] <= values[2]:
            final = values
            break
    return {
        "TemperatureData": temperature, "ConcentrationData": concentration,
        "RecordingStart_s": origin,
        "InitialConcentration": initial_concentration, "CountsData": within_window("FBRM"),
        "FlowRateData": feed, "AntisolventScaleData": scale, "AddedAntisolventData": added,
        "InitialCSDsData": seed_distribution(seed_percentiles) if seed_percentiles else EMPTY.copy(),
        "InitialD10D50D90": np.asarray(seed_percentiles or []), "FinalCSDsData": EMPTY.copy(),
        "FinalD10D50D90": np.asarray(final), **{role + "Data": data for role, data in sizes.items()},
    }
