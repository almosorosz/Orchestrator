
"""Fit a surrogate in physical recipe coordinates and propose a constrained experiment."""

from dataclasses import dataclass

import numpy as np
from scipy.special import erf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .decoder import fixed_horizon
from .physical import apply_controls, control_values, physical_profile
from .physical_fit import fit_stage_controls, initial_spec, read_spec
from .profiles import (
    CHANNELS, CONSTRAINTS, TARGETS, Profile, constraint_values, make_profile, selected_horizon, violations,
)
from .settings import OUTPUTS, STAGE_CHANNELS

POOL_SIZE = 768
STIRRING_POINTS = 8


@dataclass(frozen=True)
class Variable:
    key: str
    label: str
    unit: str


@dataclass
class Channel:
    key: str
    reference: Profile
    variables: tuple[Variable, ...]
    features: np.ndarray
    description: str
    stirring_times: np.ndarray | None = None
    stages: tuple | None = None
    stage_controls: tuple = ()
    fit_errors: np.ndarray | None = None

    def stages_at(self, values):
        return apply_controls(self.key, self.stages, self.stage_controls, values)

    def decode(self, features, durations):
        if self.stages is not None:
            return [physical_profile(self.key, self.stages_at(values), duration)
                    for values, duration in zip(features, durations)]
        if self.stirring_times is not None:
            return [Profile(self.stirring_times * duration, values, self.description, self.reference.step)
                    for duration, values in zip(durations, features)]
        if self.key == "seed":
            amounts = features[:, 0] if self.variables else np.full(len(durations), self.reference.values[0])
            return [Profile(np.array([0.]), np.array([amount]), self.description) for amount in amounts]
        if self.variables:
            return [Profile(np.array([0., duration]), np.full(2, value), self.description)
                    for duration, value in zip(durations, features[:, 0])]
        return [fixed_horizon(self.reference, duration, stop=self.key == "feed") for duration in durations]


def run_profile(run, channel):
    data = run["profiles"][channel]
    profile = make_profile(data["points"], data["source"], step=data["step"])
    if channel == "seed":
        return Profile(np.array([0.]), profile.values[:1], profile.source)
    return profile


def prepare_channel(state, runs, channel, report, *, horizon=None):
    label, unit, _ = CHANNELS[channel]
    profiles = [run_profile(run, channel) for run in runs]
    reference = profiles[0]
    if horizon is None:
        horizon = max(run["metrics"]["duration"] for run in runs)
    if channel == "temperature":
        valid = all(np.all(profile.values > -273.15) for profile in profiles)
    else:
        valid = all(np.all(profile.values >= 0) for profile in profiles)
    if not valid:
        if channel == "feed":
            raise ValueError("Antisolvent has negative recorded rates. Import the dosing rate in g/s "
                             "with pump return or other signed commands identified separately.")
        raise ValueError(f"{label} includes values outside the physical range.")
    if channel == "seed":
        values = np.array([profile.values[0] for profile in profiles])
    else:
        if any(len(profile.time) < 2 or profile.time[0] != 0
               or not np.isclose(profile.time[-1], run["metrics"]["duration"], rtol=1e-9, atol=1e-6)
               for run, profile in zip(runs, profiles)):
            raise ValueError(f"{label} must span the complete batch in each selected experiment.")
        values = np.array([profile.total() / profile.time[-1] for profile in profiles])
    if channel in STAGE_CHANNELS:
        spec = state["physical_profiles"].get(channel)
        if spec is None:
            spec = initial_spec(channel, runs)
        rows, controls = read_spec(channel, spec, horizon)
        variables = tuple(Variable(f"{channel}:{c.key}", f"{label} · {c.label}", c.unit) for c in controls)
        fitted, errors = [], []
        for index, profile in enumerate(profiles, 1):
            report(f"Fitting {label.lower()} stages · {index}/{len(profiles)}")
            point, error = fit_stage_controls(channel, rows, controls, profile, horizon)
            fitted.append(point)
            errors.append(error)
        return Channel(channel, reference, variables, np.stack(fitted), "Physical stages",
                       stages=rows, stage_controls=controls, fit_errors=np.asarray(errors))

    tolerance = state["tolerances"][channel]
    if channel == "seed" or all(np.ptp(profile.values) <= tolerance for profile in profiles):
        if np.ptp(values) > tolerance:
            return Channel(channel, reference, (Variable(f"{channel}:scalar", label, unit),),
                           values[:, None], "Scalar")
        return Channel(channel, reference, (), np.empty((len(runs), 0)), "Fixed at the first selected run")

    def same_shape(profile):
        if profile.step != reference.step:
            return False
        times = np.union1d(reference.time, profile.time)
        if profile.step:
            # Probe between knots to avoid roundoff right at a step transition.
            times = np.r_[.5 * (times[1:] + times[:-1]), 0., horizon]
        difference = reference.sample(times) - profile.sample(times)
        return np.max(np.abs(difference)) <= tolerance

    if len(runs) > 1 and all(same_shape(profile) for profile in profiles[1:]):
        return Channel(channel, reference, (), np.empty((len(runs), 0)), "Fixed at the first selected run")

    # Vary stirring directly at common batch fractions.
    times = np.linspace(0., 1., STIRRING_POINTS)
    features = np.stack([profile.sample(times * horizon) for profile in profiles])
    variables = tuple(Variable(f"{channel}:rpm{index + 1}", f"{label} at {time:.0%}", unit)
                      for index, time in enumerate(times))
    return Channel(channel, reference, variables, features, "Stirring setpoints", stirring_times=times)


def variable_activity(features):
    span = np.ptp(features, axis=0)
    return span > 1e-6 * np.maximum(1, np.max(np.abs(features), axis=0))


def input_checks(profiles, duration, constraints):
    checks = {"duration": [duration]}
    for channel, profile in profiles.items():
        key, values = constraint_values(channel, profile)
        if key:
            checks[key] = values
    return [CONSTRAINTS[key][0] for key, (lower, upper) in constraints.items()
            if (lower is not None or upper is not None)
            and (key not in checks or violations(checks[key], lower, upper))]


def metric_key(key, state):
    return f"yield_{state['yield_basis']}" if key == "yield" else key


@dataclass
class Proposal:
    state: dict
    variables: tuple[Variable, ...]
    observed: np.ndarray
    vary: np.ndarray
    scaler: StandardScaler
    surrogates: dict
    counts: dict
    unavailable: dict
    horizon: float
    point: np.ndarray
    controls: np.ndarray
    profiles: dict
    stages: dict
    predictions: dict
    mean_loss: float
    loss_sd: float
    acquisition: float
    input_passed: int
    output_passed: int
    channels: dict

    def projection(self, dimension, output):
        """Vary one coordinate and hold the remaining coordinates at the proposal."""
        center = self.point[dimension]
        x = np.linspace(min(self.observed[:, dimension].min(), center),
                        max(self.observed[:, dimension].max(), center), 81)
        points = np.tile(self.point, (len(x), 1))
        points[:, dimension] = x
        if output == "loss":
            mean, sd = objective_loss(self.predict(points), self.state)
        else:
            mean, sd = self.predict(points)[output]
        return x, mean, sd

    def predict(self, points):
        return predictions(points, self.scaler, self.surrogates, self.horizon)

    def observations(self, output):
        runs = [self.state["runs"][index] for index in self.state["selected"]]
        if output != "loss":
            return np.array([run["metrics"][metric_key(output, self.state)] for run in runs], dtype=float)
        outcomes = {key: (np.array([run["metrics"][metric_key(key, self.state)] for run in runs], dtype=float),
                          np.zeros(len(runs))) for key, target in self.state["targets"].items()
                    if key == "duration" or (target["enabled"] and target["weight"] > 0)}
        complete = np.logical_and.reduce([np.isfinite(values) for values, _ in outcomes.values()])
        measured = np.full(len(runs), np.nan)
        if complete.any():
            valid = {key: (values[complete], std[complete]) for key, (values, std) in outcomes.items()}
            measured[complete] = objective_loss(valid, self.state)[0]
        return measured


def check_settings(state, channels):
    needed = {"cooling": "temperature", "seed": "seed", "feed": "feed"}
    for key, channel in needed.items():
        if any(limit is not None for limit in state["constraints"][key]) and channel not in channels:
            raise ValueError(f"{CONSTRAINTS[key][0]} bounds need {CHANNELS[channel][0].lower()} "
                             "data in the selected experiments.")
    has_target = any(state["targets"][key]["enabled"] and state["targets"][key]["weight"] > 0
                     for key in OUTPUTS)
    has_output_bounds = any(value is not None for bounds in state["output_bounds"].values() for value in bounds)
    if not has_target and not has_output_bounds:
        raise ValueError("Enable an output target or output limit. Batch duration is fixed.")


def fit_surrogates(runs, state, design, report):
    fitted, counts, unavailable = {}, {"duration": len(runs)}, {}
    for key in OUTPUTS:
        responses = [run["metrics"][metric_key(key, state)] for run in runs]
        valid = np.array([value is not None for value in responses])
        y = np.asarray(responses, dtype=float)[valid]
        counts[key] = len(y)
        if len(y) < 3:
            unavailable[key] = f"Needs 3 measured values for {TARGETS[key][0]} ({len(y)} available)"
        elif np.ptp(y) <= 1e-12 or len(np.unique(design[valid], axis=0)) < 2:
            unavailable[key] = "Needs variation in both the response and input recipes"
        else:
            report(f"Fitting the {TARGETS[key][0]} Gaussian process · {len(y)} observations")
            gp = GaussianProcessRegressor(kernel=Matern(length_scale=2., nu=2.5) + WhiteKernel(.02),
                                          optimizer=None, normalize_y=True)
            fitted[key] = gp.fit(design[valid], y)

        target = state["targets"][key]
        required = ((target["enabled"] and target["weight"] > 0)
                    or any(value is not None for value in state["output_bounds"][key]))
        if required and key in unavailable:
            raise ValueError(f"{TARGETS[key][0]} has no usable surrogate. {unavailable[key]}.")
    return fitted, counts, unavailable


def predictions(points, scaler, surrogates, horizon):
    scaled = scaler.transform(points)
    outcomes = {key: process.predict(scaled, return_std=True) for key, process in surrogates.items()}
    outcomes["duration"] = (np.full(len(points), horizon), np.zeros(len(points)))
    return outcomes


def objective_loss(outcomes, state):
    mean_loss = np.zeros_like(outcomes["duration"][0])
    variance = np.zeros_like(mean_loss)
    with np.errstate(over="ignore", invalid="ignore"):
        for key, target in state["targets"].items():
            if key == "duration" or not target["enabled"] or not target["weight"]:
                continue
            mean, sd = outcomes[key]
            if target["mode"] == "max":
                mean = -mean
            elif target["mode"] == "match":
                distance = np.abs(mean - target["target"])
                ratio = np.divide(distance, sd, where=sd > 0, out=np.full_like(distance, np.inf))
                mean = distance * erf(ratio / np.sqrt(2)) + sd * np.sqrt(2 / np.pi) * np.exp(-.5 * ratio**2)
                sd = np.sqrt(np.maximum(distance**2 + sd**2 - mean**2, 0.))
            factor = target["weight"] / target["scale"]
            mean_loss += factor * mean
            variance += (factor * sd)**2
    if not np.isfinite(mean_loss).all() or not np.isfinite(variance).all():
        raise ValueError("The objective could not be scaled. Check the target weights and scales.")
    return mean_loss, np.sqrt(variance)


def allowed_outputs(outcomes, state):
    valid = np.ones(len(outcomes["duration"][0]), dtype=bool)
    margin = state["output_margin"]
    for key, (lower, upper) in state["output_bounds"].items():
        if lower is None and upper is None:
            continue
        mean, std = outcomes[key]
        if lower is not None:
            valid &= mean - margin * std >= lower
        if upper is not None:
            valid &= mean + margin * std <= upper
    sizes = [outcomes[key][0] for key in ("d10", "d50", "d90") if key in outcomes]
    for lower, upper in zip(sizes, sizes[1:]):
        valid &= lower <= upper
    return valid


def decode_recipe(channels, slices, point, horizon):
    recipe = {}
    for key, channel in channels.items():
        columns = slices[key]
        recipe[key], = channel.decode(point[None, columns], [horizon])
    return recipe


def propose(state, report=lambda _: None):
    runs = [state["runs"][index] for index in state["selected"]]
    if len(runs) < 2:
        raise ValueError("Select at least two completed experiments.")
    horizon = selected_horizon(state)
    if violations([horizon], *state["constraints"]["duration"]):
        raise ValueError(f"The fixed batch horizon ({horizon:g} s) is outside the batch duration bounds. "
                         "Change the bounds or select a different set of experiments.")
    channels, column_slices = {}, {}
    variables, columns = [], []
    for key, (label, _, _) in CHANNELS.items():
        has_channel = [key in run["profiles"] for run in runs]
        if not any(has_channel):
            if key in state["physical_profiles"]:
                raise ValueError(f"The selected runs have no recorded {label.lower()} profile for fitting.")
            continue
        if not all(has_channel):
            raise ValueError(f"{label} data is missing from some selected runs. Select runs with "
                             "matching profile coverage or complete the missing profiles in IMED.")
        report(f"Preparing {label.lower()} inputs")
        channel = channels[key] = prepare_channel(state, runs, key, report, horizon=horizon)
        start = len(variables)
        variables.extend(channel.variables)
        columns.append(channel.features)
        column_slices[key] = slice(start, len(variables))
    check_settings(state, channels)
    observed = np.column_stack(columns)
    active = variable_activity(observed)
    vary = active & np.array([state["overrides"].get(variable.key, True) for variable in variables], dtype=bool)
    if not vary.any():
        raise ValueError("Choose at least one stage control, setpoint or seed mass that varies across the runs.")
    # Keep measured variation in the GP even when a decision variable is held fixed for the search.
    scaler = StandardScaler().fit(observed)
    scaler.scale_[~active] = 1.
    scaled = scaler.transform(observed)
    surrogates, counts, unavailable = fit_surrogates(runs, state, scaled, report)

    report(f"Decoding and checking {POOL_SIZE} interpolated recipes")
    rng = np.random.default_rng(42)
    left = rng.integers(len(runs), size=POOL_SIZE)
    right = (left + rng.integers(1, len(runs), size=POOL_SIZE)) % len(runs)
    portion = rng.uniform(.02, .98, (POOL_SIZE, 1))
    points = observed[left] * portion + observed[right] * (1 - portion)
    points[:, ~vary] = observed[0, ~vary]
    for key, channel in channels.items():
        if channel.stages is not None:
            columns = column_slices[key]
            local = points[:, columns]
            local[:, ~vary[columns]] = control_values(channel.stages, channel.stage_controls)[~vary[columns]]
    seed_column = next((col for col, var in enumerate(variables) if var.key == "seed:scalar"), None)
    if seed_column is not None and vary[seed_column]:
        # The zero-mass mode and positive-mass mode come from executed runs.
        # Never interpolate a new seed mass between an unseeded and a seeded recipe.
        before, after = observed[left, seed_column], observed[right, seed_column]
        mixed = (before == 0.) != (after == 0.)
        points[mixed, seed_column] = np.where(portion[mixed, 0] >= .5, before[mixed], after[mixed])
    accepted, rejected = [], {}
    for row, point in enumerate(points):
        try:
            recipe = decode_recipe(channels, column_slices, point, horizon)
        except ValueError:
            rejected["Stage duration or physical setting"] = None
            continue
        failed = input_checks(recipe, horizon, state["constraints"])
        if failed:
            rejected.update(dict.fromkeys(failed))
        else:
            accepted.append(row)
    if not accepted:
        failed = ", ".join(rejected)
        raise ValueError(f"None of {POOL_SIZE} decoded recipes met the input bounds ({failed}).")
    kept = np.asarray(accepted)
    evaluated = points[kept]

    evaluated_scaled = scaler.transform(evaluated)
    duplicate = np.array([np.any(np.max(np.abs(scaled - point), axis=1) <= 1e-6)
                          for point in evaluated_scaled])
    # Require the decoded point to be within one training-neighborhood radius.
    # This avoids selecting poorly reconstructed profiles merely for their high extrapolation uncertainty.
    nearest = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=2)
    nearest[nearest <= 1e-8] = np.inf
    spacing = np.min(nearest, axis=1)
    radius = float(np.max(spacing[np.isfinite(spacing)], initial=0.))
    distances = np.linalg.norm(evaluated_scaled[:, None, :] - scaled[None, :, :], axis=2)
    in_domain = np.min(distances, axis=1) <= radius + 1e-10
    new_recipe = ~duplicate & in_domain
    if not new_recipe.any():
        raise ValueError("The decoded recipes duplicated selected runs or fell outside the observed profile "
                         "neighborhood. Review the selected controls and fitted traces in Physical profiles.")
    kept, evaluated = kept[new_recipe], evaluated[new_recipe]

    report(f"Predicting outputs and screening {len(kept)} recipes")
    outcomes = predictions(evaluated, scaler, surrogates, horizon)
    valid_output = allowed_outputs(outcomes, state)
    if not valid_output.any():
        raise ValueError(f"None of {len(kept)} recipes passed the output screening. "
                         "Review the output limits and margin.")
    mean, sd = objective_loss(outcomes, state)
    acquisition = mean - state["exploration"] * sd
    if not np.isfinite(acquisition).all():
        raise ValueError("The acquisition could not be calculated. Check the exploration weight.")
    best = int(np.argmin(np.where(valid_output, acquisition, np.inf)))
    responses = {key: (float(mu[best]), float(sigma[best])) for key, (mu, sigma) in outcomes.items()}
    control = points[kept[best]]
    recipe = decode_recipe(channels, column_slices, control, horizon)
    stages = {key: channel.stages_at(control[column_slices[key]])
              for key, channel in channels.items() if channel.stages is not None}
    return Proposal(state, tuple(variables), observed, vary, scaler, surrogates, counts, unavailable, horizon,
                    evaluated[best], control, recipe, stages, responses, float(mean[best]), float(sd[best]),
                    float(acquisition[best]), len(kept), int(valid_output.sum()), channels)


def preview_channel(state, key, index, report):
    runs = [state["runs"][i] for i in state["selected"] if key in state["runs"][i]["profiles"]]
    run = state["runs"][index]
    horizon = selected_horizon(state)
    channel = prepare_channel(state, runs, key, report, horizon=horizon)
    return channel, runs, next(i for i, peer in enumerate(runs) if peer is run)
