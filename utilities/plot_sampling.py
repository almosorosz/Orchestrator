"""Bounded, extrema-preserving plot reduction and event-aware simulation ticks."""
import math
import numpy as np


def extrema_indices(series, count, budget, mandatory=()):
    if count <= budget:
        return list(range(count))
    keep = {0, count - 1, *mandatory}
    if len(keep) > budget:
        ordered = sorted(keep)
        return [ordered[i] for i in np.linspace(0, len(ordered) - 1, budget, dtype=int)]
    # Reserve four extrema per channel (first/last occurrence of min/max),
    # plus each bucket's endpoints. Plateau widths survive alongside peaks.
    arrays = [np.asarray(values, dtype=float) for values in series]
    buckets = max(1, (budget - len(keep)) // max(2, 2 + 4 * len(arrays)))
    for indices in np.array_split(np.arange(1, count - 1), buckets):
        if not len(indices):
            continue
        candidates = {int(indices[0]), int(indices[-1])}
        for values in arrays:
            valid = indices[np.isfinite(values[indices])]
            if not len(valid):
                continue
            sample = values[valid]
            for value in (sample.min(), sample.max()):
                matches = valid[sample == value]
                candidates.update((int(matches[0]), int(matches[-1])))
        keep.update(sorted(candidates)[:max(0, budget - len(keep))])
    return sorted(keep)


def simulation_ticks(frame, poll_interval, end_hold_s):
    """Include every waypoint and at least eight intervals on changing ramps."""
    times = frame["time_s"].to_numpy(dtype=float)
    targets = frame.drop(columns="time_s").to_numpy(dtype=float)
    yield 0., True
    left = 0.
    for index, right in enumerate(times):
        if right <= left:
            continue
        changing = index > 0 and not np.array_equal(targets[index - 1], targets[index])
        intervals = max(8 if changing else 1, math.ceil((right - left) / poll_interval))
        for step in range(1, intervals + 1):
            yield (float(right) if step == intervals else left + (right - left) * step / intervals), step == intervals
        left = float(right)
    if end_hold_s:
        intervals = max(1, math.ceil(end_hold_s / poll_interval))
        for step in range(1, intervals + 1):
            yield left + end_hold_s * step / intervals, step == intervals
