"""Extend recorded profiles and measure their timing."""

import numpy as np

from .profiles import Profile


def fixed_horizon(profile, horizon, *, stop=False):
    """Extend a trace in seconds, holding its final value or stopping feed."""
    time, values = profile.time, profile.values
    end = time[-1]
    if horizon <= end:
        return profile
    if stop and values[-1] != 0:
        if profile.step:
            values = np.r_[values[:-1], 0.]
        else:
            # End a recorded linear feed at its last sample without adding any dose.
            after = np.nextafter(end, horizon)
            if after < horizon:
                time, values = np.r_[time, after], np.r_[values, 0.]
    return Profile(np.r_[time, horizon], np.r_[values, 0. if stop else values[-1]],
                   profile.source, profile.step)


def cooling_landmarks(profile):
    """Times of 10%, 50% and 90% of the drop from the initial temperature."""
    drop = np.maximum.accumulate(np.maximum(profile.values[0] - profile.values, 0.))
    if drop[-1] <= 1e-6:
        return None
    levels = drop[-1] * np.array([.10, .50, .90])
    high = np.searchsorted(drop, levels)
    if profile.step:
        return profile.time[high]
    left, right = profile.values[0] - profile.values[high - 1], drop[high]
    fraction = (levels - left) / (right - left)
    return profile.time[high - 1] + fraction * (profile.time[high] - profile.time[high - 1])


def dose_landmarks(profile):
    time, rate = profile.time, profile.values
    right = rate[:-1] if profile.step else rate[1:]
    mass = np.r_[0., np.cumsum(np.diff(time) * (rate[:-1] + right) / 2)]
    if mass[-1] <= 0:
        return None
    levels = mass[-1] * np.array([.05, .50, .95])
    high = np.searchsorted(mass, levels)
    # Solve the integrated linear rate within the crossing interval.
    left_rate, right_rate = rate[high - 1], right[high - 1]
    width = time[high] - time[high - 1]
    rate_area = (levels - mass[high - 1]) / width
    root = np.sqrt(np.maximum(0., left_rate**2 + 2 * (right_rate - left_rate) * rate_area))
    fraction = np.clip(2 * rate_area / (left_rate + root), 0., 1.)
    return time[high - 1] + fraction * width
