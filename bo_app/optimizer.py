"""Select the next experiment from generated CrystalLoop recipes."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .vae import encode_profiles


@dataclass
class Campaign:
    profiles: pd.DataFrame
    outcomes: pd.DataFrame
    features: np.ndarray
    planned_time: np.ndarray


def load_campaign(profile_path, outcome_path):
    profiles = pd.read_csv(profile_path)
    outcomes = pd.read_csv(outcome_path).set_index("experiment", verify_integrity=True)
    names, temperatures, antisolvent, durations = [], [], [], []
    for name, group in profiles.groupby("experiment", sort=False):
        data = group[["time_s", "T", "AS"]].to_numpy(float)
        if len(data) < 2 or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError(f"{name} requires finite profiles with strictly increasing time")
        duration = data[-1, 0] - data[0, 0]
        sample_time = np.linspace(0, 1, 8)
        time = (data[:, 0] - data[0, 0]) / duration
        names.append(name)
        durations.append(duration)
        temperatures.append(np.interp(sample_time, time, data[:, 1]))
        antisolvent.append(np.interp(sample_time, time, data[:, 2]))
    if len(names) < 2 or set(names) != set(outcomes.index):
        raise ValueError("Provide one outcome or candidate row per generated experiment")
    outcomes = outcomes.loc[names]
    amounts = outcomes["seed_amount"].to_numpy(float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Every experiment needs a finite nonnegative seed amount")
    columns = []
    for index, values in enumerate((temperatures, antisolvent)):
        values = np.asarray(values)
        values = (values - values.min()) / (np.ptp(values) or 1)
        normalized_time = np.broadcast_to(sample_time, values.shape)
        columns.append(encode_profiles(np.stack((normalized_time, values), axis=1), seed=42 + index))
    features = np.column_stack((*columns, amounts, durations))
    return Campaign(profiles, outcomes, StandardScaler().fit_transform(features), np.asarray(durations))


def rank_candidates(campaign, size_mode, target, weights, exploration):
    data = campaign.outcomes
    states = data["status"].str.strip().str.lower()
    if not states.isin(["completed", "candidate", "pending", "excluded"]).all():
        raise ValueError("Use status completed, candidate, pending, or excluded")
    completed = states.eq("completed").to_numpy()
    candidate = states.eq("candidate").to_numpy()
    weights = np.asarray(weights, dtype=float)
    if not completed.any() or not candidate.any():
        raise ValueError("At least one completed experiment and one candidate are required")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Give at least one objective a positive weight")

    def column(name):
        return data.get(name, pd.Series(np.nan, index=data.index)).to_numpy(float)

    size = column("size")
    if size_mode == "Maximize":
        size = -size
    elif size_mode == "Target":
        size = np.abs(size - target)
    initial = column("c_initial")
    measured = column("c_final")
    equilibrium = column("c_equilibrium")
    # All concentration fields use a common mass basis.
    final = np.where(np.isfinite(measured), measured, equilibrium)
    yield_fraction = np.full(len(data), np.nan)
    np.divide(initial - final, initial, out=yield_fraction, where=initial > 0)
    batch_time = column("batch_time_s")
    batch_time = np.where(np.isfinite(batch_time), batch_time, campaign.planned_time)

    observed = np.column_stack((size, -yield_fraction, batch_time))[:, weights > 0]
    weights = weights[weights > 0]
    valid = completed & np.isfinite(observed).all(axis=1)
    if not valid.any():
        raise ValueError("No completed experiment has data for every enabled objective")
    standardized = StandardScaler().fit_transform(observed[valid])
    loss = standardized @ (weights / weights.sum())
    gp = GaussianProcessRegressor(kernel=Matern(length_scale=2, nu=2.5) + WhiteKernel(0.02),
                                  optimizer=None, normalize_y=True)
    gp.fit(campaign.features[valid], loss)
    mean, std = gp.predict(campaign.features[candidate], return_std=True)
    names = data.index[candidate]
    return pd.DataFrame({"experiment": names, "predicted_loss": mean, "uncertainty": std,
                         "score": mean - exploration * std}).sort_values("score")
