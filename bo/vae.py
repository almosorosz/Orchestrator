
"""Inference for the pretrained temperature and antisolvent trajectory VAEs."""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.paths import ASSETS_DIR

from .profiles import CHANNELS, Profile
# Compatibility for the original standalone VAE tooling.
CHECKPOINT_CHANNELS = {"temperature": "T", "feed": "AS"}
LATENT_DIM = 4


def checkpoint_filename(channel):
    return f"trajectory_vae_{CHECKPOINT_CHANNELS[channel]}.pt"


SEQ_LEN = 8
SUPPORTED_SEQUENCE_LENGTHS = (8, 10)
MODEL_LOAD_LOCK = Lock()


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


def _checkpoint_model(channel, saved):
    kind = CHECKPOINT_CHANNELS[channel]
    if not isinstance(saved, dict) or not isinstance(saved.get("model_state_dict"), dict):
        raise ValueError("Choose a trajectory VAE checkpoint exported by the notebook.")
    if saved.get("profile_kind") != kind:
        raise ValueError(f"Select the notebook checkpoint for {kind}.")
    latent_dim = saved.get("latent_dim")
    if type(latent_dim) is not int or latent_dim != LATENT_DIM:
        raise ValueError(f"The {kind} checkpoint must have {LATENT_DIM} latent variables (found {latent_dim}).")
    count = saved.get("seq_len")
    if (type(count) is not int or count not in SUPPORTED_SEQUENCE_LENGTHS
            or saved.get("hidden_dims") not in ([64, 32], (64, 32))
            or saved.get("num_bands") != 2):
        raise ValueError("This checkpoint uses an unsupported trajectory VAE architecture.")
    # Instantiating a model consumes RNG state even when all weights are immediately replaced.
    with MODEL_LOAD_LOCK, torch.random.fork_rng(devices=[]):
        model = ImprovedTrajectoryVAE(seq_len=count)
    weights = saved["model_state_dict"]
    expected = model.state_dict()
    if weights.keys() != expected.keys():
        raise ValueError("The checkpoint weights do not match the trajectory VAE.")
    state = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in weights.items()}
    if any(state[key].shape != value.shape or not torch.isfinite(state[key]).all()
           for key, value in expected.items()):
        raise ValueError("The checkpoint contains invalid trajectory VAE weights.")
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)


def read_checkpoint(channel, path):
    """Import the notebook's weights-only .pt file as portable JSON data."""
    model = _checkpoint_model(channel, torch.load(path, map_location="cpu", weights_only=True))
    return {"name": Path(path).name, "profile_kind": CHECKPOINT_CHANNELS[channel], "seq_len": model.seq_len,
            "latent_dim": model.latent_dim, "hidden_dims": [64, 32], "num_bands": 2,
            "model_state_dict": {key: value.tolist() for key, value in model.state_dict().items()}}


@lru_cache(maxsize=len(CHECKPOINT_CHANNELS))
def bundled_model(channel, path):
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError as error:
        raise ValueError(f"Pretrained weights are missing for {CHANNELS[channel][0]}. "
                         f"Check the legacy model file {path.name}.") from error
    return _checkpoint_model(channel, saved)


def sample_knots(profile, count=SEQ_LEN):
    """Keep endpoints and select knots around the largest trace deviations."""
    time, values = profile.time, profile.values
    if len(time) > count:
        knots = {0, len(time) - 1}
        while len(knots) < count:
            indices = sorted(knots)
            errors = []
            for first, last in zip(indices[:-1], indices[1:]):
                if last - first < 2:
                    continue
                segment = np.arange(first + 1, last)
                slope = values[last] - values[first]
                duration = time[last] - time[first]
                predicted = values[first] + slope * (time[segment] - time[first]) / duration
                gap = np.abs(values[segment] - predicted)
                index = int(np.argmax(gap)) if gap.max() > 1e-12 else len(segment) // 2
                errors.append((float(gap[index]), duration, int(segment[index])))
            knots.add(max(errors)[2])
        indices = sorted(knots)
        return np.stack((time[indices], values[indices]))
    while len(time) < count:
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
        duration = profile.time[-1]
        scale = self.scale or 1.
        normalized = Profile(profile.time / duration, (profile.values - self.offset) / scale,
                             profile.source, profile.step)
        return sample_knots(normalized, self.model.seq_len)

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


def pretrained_codec(channel, profiles, checkpoint):
    path = ASSETS_DIR / "bo" / checkpoint_filename(channel)
    if checkpoint is None:
        model, name = bundled_model(channel, path), path.name
    else:
        model, name = _checkpoint_model(channel, checkpoint), checkpoint["name"]
    # The notebooks operate in [0, 1]. Use one physical range across the selected
    # recipes, with zero feed mapped to zero. The weights are shared across campaigns.
    lower = 0. if channel == "feed" else min(float(profile.values.min()) for profile in profiles)
    upper = max(float(profile.values.max()) for profile in profiles)
    return TrajectoryCodec(model, lower, upper - lower, all(profile.step for profile in profiles),
                           f"{name} · pretrained")
