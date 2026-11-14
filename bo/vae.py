
"""The trajectory VAE from BO_test_V2_single_profile, with notebook checkpoint support."""

from dataclasses import dataclass
import math
from threading import Lock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .profiles import Profile

SEQ_LEN = 8
LATENT_DIM = 7
CHECKPOINT_KINDS = {"temperature": "T", "feed": "AS"}
FIT_LOCK = Lock()


class PositionalEncoding1D(nn.Module):
    def __init__(self, num_bands=2):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, x):
        features = [x]
        for band in range(self.num_bands):
            angle = (2.0 ** band) * math.pi * x
            features.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(features, dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + 0.5 * self.net(x)


class ImprovedTrajectoryVAE(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, latent_dim=LATENT_DIM, hidden_dims=(64, 32), num_bands=2):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        h1, h2 = hidden_dims
        self.pos_enc = PositionalEncoding1D(num_bands)
        input_dim = 2 * seq_len * (1 + 2 * num_bands)
        self.encoder = nn.Sequential(nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.SiLU(), ResidualBlock(h1),
                                     nn.Linear(h1, h2), nn.LayerNorm(h2), nn.SiLU())
        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, h2), nn.SiLU(), nn.Linear(h2, h1), nn.SiLU(), ResidualBlock(h1))
        self.decoder_profile = nn.Sequential(nn.Linear(h1, h2), nn.SiLU(), nn.Linear(h2, seq_len), nn.Sigmoid())
        self.decoder_t_profile = nn.Sequential(nn.Linear(h1, h2), nn.SiLU(), nn.Linear(h2, seq_len - 1))

    def encode(self, x):
        batch, channels, length = x.shape
        if channels != 2 or length != self.seq_len:
            raise ValueError(f"Expected profiles with shape (batch, 2, {self.seq_len}).")
        features = self.pos_enc(x.reshape(batch * channels, length)).reshape(batch, -1)
        hidden = self.encoder(features)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def decode(self, z):
        hidden = self.decoder(z)
        values = self.decoder_profile(hidden)
        intervals = F.softplus(self.decoder_t_profile(hidden)) + 1e-5
        intervals = intervals / intervals.sum(dim=-1, keepdim=True)
        time = torch.cat((torch.zeros_like(intervals[:, :1]), intervals.cumsum(dim=-1)), dim=-1)
        return torch.stack((time, values), dim=1)

    def forward(self, x):
        mean, log_variance = self.encode(x)
        latent = mean
        if self.training:
            std = (0.5 * log_variance.clamp(-12, 8)).exp()
            latent = mean + torch.randn_like(std) * std
        return self.decode(latent), mean, log_variance


def trajectory_loss(reconstructed, target, mean, log_variance, beta, *, dosing=False):
    """Use the notebook loss, including its switch, endpoint and free-bits terms."""
    time, values = target.unbind(dim=1)
    predicted_time, predicted_values = reconstructed.unbind(dim=1)
    dp, predicted_dp = values.diff(), predicted_values.diff()
    dt, predicted_dt = time.diff(), predicted_time.diff()
    weights = 1 + 3 * dp.abs()

    def weighted(error, weight):
        return ((error * weight).sum(-1) / weight.sum(-1).clamp_min(1e-8)).mean()

    loss = F.smooth_l1_loss(predicted_time, time, beta=.05)
    amplitude_weight = 1. if dosing else 1.5
    loss += amplitude_weight * F.smooth_l1_loss(predicted_values, values, beta=.05)
    interval_error = weighted((predicted_dt - dt).abs(), weights)
    interval_error += .1 * weighted(F.smooth_l1_loss(predicted_dt.clamp_min(1e-6).log(),
                                                      dt.clamp_min(1e-6).log(), beta=.5,
                                                      reduction="none"), weights)
    loss += (.02 if dosing else .01) * interval_error
    loss += (.5 if dosing else .1) * weighted((predicted_dp - dp).abs(), weights)
    midpoint_error = .5 * (predicted_time[:, :-1] + predicted_time[:, 1:] - time[:, :-1] - time[:, 1:])
    loss += (2. if dosing else 1.) * weighted(midpoint_error.abs(), dp.abs())
    if not dosing:
        loss += .1 * F.l1_loss(predicted_dp.diff(), dp.diff())
    loss += 2 * amplitude_weight * F.mse_loss(predicted_values[:, [0, -1]], values[:, [0, -1]]) * 2
    log_variance = log_variance.clamp(-12, 8)
    kl = -.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean(dim=0)
    return loss + beta * kl.clamp_min(.05).sum()


def fit_vae(trajectories, *, channel, seed, report):
    """Fit the notebook architecture to the selected physical profiles."""
    targets = torch.as_tensor(np.asarray(trajectories), dtype=torch.float32)
    epochs = 300
    with FIT_LOCK, torch.random.fork_rng(devices=[]):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(seed)
            model = ImprovedTrajectoryVAE()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(epochs):
                for indices in torch.randperm(len(targets)).split(64):
                    batch = targets[indices]
                    reconstructed, mean, log_variance = model(batch)
                    loss = trajectory_loss(reconstructed, batch, mean, log_variance,
                                           1e-3 * min(1., (epoch + 1) / 30), dosing=channel == "feed")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    optimizer.step()
                if epoch == 0 or (epoch + 1) % 100 == 0:
                    report(f"Fitting the {channel} VAE · epoch {epoch + 1}/{epochs}")
        finally:
            torch.set_num_threads(previous_threads)
    return model.eval()


def _checkpoint_model(channel, saved):
    if saved.get("profile_kind") != CHECKPOINT_KINDS[channel]:
        raise ValueError(f"Select the notebook checkpoint for {CHECKPOINT_KINDS[channel]}.")
    latent_dim = saved.get("latent_dim")
    if (saved.get("seq_len") != SEQ_LEN or type(latent_dim) is not int or not 1 <= latent_dim <= 32
            or list(saved.get("hidden_dims", ())) != [64, 32] or saved.get("num_bands") != 2):
        raise ValueError("This checkpoint uses an unsupported trajectory VAE architecture.")
    # Instantiating a model consumes RNG state even when all weights are immediately replaced.
    with FIT_LOCK, torch.random.fork_rng(devices=[]):
        model = ImprovedTrajectoryVAE(latent_dim=latent_dim)
    weights = saved["model_state_dict"]
    expected = model.state_dict()
    if weights.keys() != expected.keys():
        raise ValueError("The checkpoint weights do not match the trajectory VAE.")
    state = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in weights.items()}
    if any(state[key].shape != value.shape or not torch.isfinite(state[key]).all()
           for key, value in expected.items()):
        raise ValueError("The checkpoint contains invalid trajectory VAE weights.")
    model.load_state_dict(state)
    return model.eval()


def read_checkpoint(channel, path):
    """Import the notebook's weights-only .pt file as portable JSON data."""
    from pathlib import Path

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("model_state_dict"), dict):
        raise ValueError("Choose a trajectory VAE checkpoint exported by the notebook.")
    model = _checkpoint_model(channel, saved)
    return {"name": Path(path).name, "profile_kind": CHECKPOINT_KINDS[channel], "seq_len": SEQ_LEN,
            "latent_dim": model.latent_dim, "hidden_dims": [64, 32], "num_bands": 2,
            "model_state_dict": {key: value.tolist() for key, value in model.state_dict().items()}}


def sample_knots(profile):
    """Keep endpoints and select eight knots around the largest trace deviations."""
    time, values = profile.time, profile.values
    if len(time) > SEQ_LEN:
        knots = {0, len(time) - 1}
        while len(knots) < SEQ_LEN:
            spans = zip(sorted(knots)[:-1], sorted(knots)[1:])
            errors = []
            for first, last in spans:
                if last - first < 2:
                    continue
                segment = slice(first + 1, last)
                predicted = np.interp(time[segment], time[[first, last]], values[[first, last]])
                gap = np.abs(values[segment] - predicted)
                index = int(np.argmax(gap)) if gap.max() > 0 else (last - first) // 2 - 1
                errors.append((float(gap[index]), time[last] - time[first], first + 1 + index))
            knots.add(max(errors)[2])
        indices = sorted(knots)
        return np.stack((time[indices], values[indices]))
    while len(time) < SEQ_LEN:
        split = int(np.argmax(np.diff(time))) + 1
        midpoint = (time[split - 1] + time[split]) / 2
        time = np.insert(time, split, midpoint)
        values = np.insert(values, split, profile.sample(midpoint))
    return np.stack((time, values))


@dataclass
class TrajectoryCodec:
    model: ImprovedTrajectoryVAE
    offset: float
    scale: float
    step: bool
    description: str

    def normalize(self, profile):
        time, values = sample_knots(profile)
        return np.stack((time / time[-1], (values - self.offset) / self.scale))

    def encode(self, profiles):
        inputs = torch.as_tensor(np.stack([self.normalize(profile) for profile in profiles]), dtype=torch.float32)
        with torch.inference_mode():
            mean, _ = self.model.encode(inputs)
        return mean.numpy().astype(float)

    def decode(self, features, durations):
        with torch.inference_mode():
            trajectories = self.model.decode(torch.as_tensor(features, dtype=torch.float32)).numpy().astype(float)
        profiles = []
        for duration, (time, values) in zip(durations, trajectories):
            time = time / time[-1] * duration
            values = values * self.scale + self.offset
            profiles.append(Profile(time, values, "VAE decoded", self.step))
        return profiles


def fit_codec(channel, profiles, checkpoint, report, seed):
    lower = min(float(profile.values.min()) for profile in profiles)
    upper = max(float(profile.values.max()) for profile in profiles)
    codec = TrajectoryCodec(None, lower, upper - lower or 1.,
                            all(profile.step for profile in profiles), "")
    if checkpoint is not None:
        codec.model = _checkpoint_model(channel, checkpoint)
        codec.description = checkpoint["name"]
    else:
        codec.model = fit_vae([codec.normalize(profile) for profile in profiles],
                              channel=channel, seed=seed, report=report)
        codec.description = "Notebook architecture fitted to selected experiments"
    return codec
