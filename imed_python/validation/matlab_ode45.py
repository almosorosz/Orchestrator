"""MATLAB-style Dormand–Prince stepping for archived IMED comparisons."""

import numpy as np
from scipy.integrate import RK45


# Polynomial coefficients for the continuous extension used by ode45.
# The stages are the same Dormand–Prince pair available through SciPy RK45.
_DENSE = np.array([
    [1, -183 / 64, 37 / 12, -145 / 128],
    [0, 0, 0, 0],
    [0, 1500 / 371, -1000 / 159, 1000 / 371],
    [0, -125 / 32, 125 / 12, -375 / 64],
    [0, 9477 / 3392, -729 / 106, 25515 / 6784],
    [0, -11 / 7, 11 / 3, -55 / 28],
    [0, 3 / 2, -4, 5 / 2],
])


def integrate_matlab_ode45(rhs, time_points, initial, *, rtol=1e-3, atol=1e-6, max_step=None):
    """Integrate a forward ODE on the requested times with ode45-style controls."""
    times = np.asarray(time_points, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Integration output times must be strictly increasing")
    if max_step is not None and not max_step > 0:
        raise ValueError("MaxStep must be positive")

    time, end = times[[0, -1]]
    step_limit = min(max_step if max_step is not None else (end - time) / 10, end - time)
    state = np.asarray(initial, dtype=float)
    stages = np.empty((7, len(state)))
    states = np.empty((len(state), len(times)))
    states[:, 0] = state
    stages[0] = rhs(time, state)
    initial_rate = np.max(np.abs(stages[0]) / np.maximum(np.abs(state), atol / rtol))
    step = min(step_limit, times[1] - time)
    if initial_rate:
        step = min(step, 0.8 * rtol**0.2 / initial_rate)

    next_output = 1
    while time < end:
        minimum = 16 * np.spacing(abs(time))
        step = min(step_limit, max(step, minimum))
        if end - time <= 1.1 * step:
            step = end - time

        attempts = 0
        while True:
            for index in range(1, 6):
                stage_state = state + step * (RK45.A[index, :index] @ stages[:index])
                stages[index] = rhs(time + step * RK45.C[index], stage_state)
            next_time = min(time + step, end)
            step = next_time - time
            next_state = state + step * (RK45.B @ stages[:6])
            stages[-1] = rhs(next_time, next_state)
            scale = np.maximum(atol, rtol * np.maximum(np.abs(state), np.abs(next_state)))
            error = np.max(np.abs(step * (RK45.E @ stages)) / scale)
            if np.isfinite(error) and error <= 1:
                break
            if step <= minimum:
                raise RuntimeError(f"Integration tolerances could not be met at {time:g} s")
            factor = max(0.1, 0.8 * error**-0.2) if attempts == 0 else 0.5
            step = max(step * factor, minimum)
            attempts += 1

        output_end = np.searchsorted(times, next_time, side="right")
        if output_end > next_output:
            fractions = (times[next_output:output_end] - time) / step
            powers = fractions ** np.arange(1, 5)[:, None]
            states[:, next_output:output_end] = state[:, None] + step * (stages.T @ _DENSE) @ powers
            if times[output_end - 1] == next_time:
                states[:, output_end - 1] = next_state
            next_output = output_end

        time, state = next_time, next_state
        stages[0] = stages[-1]
        if not attempts:
            step *= min(5.0, 0.8 * error**-0.2) if error else 5.0

    if next_output != len(times):
        raise RuntimeError("Integration ended before filling the requested output times")
    return states
