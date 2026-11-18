"""Trajectory VAE architecture from the IMED single-profile BO notebooks."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# 1. Fourier Feature Encoding
class PositionalEncoding1D(nn.Module):
    def __init__(self, num_bands=2):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, x):
        # x: (batch, seq_len)
        out = [x]
        for i in range(self.num_bands):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


# 2. Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)


# 3. Single-profile Trajectory VAE
class ImprovedTrajectoryVAE(nn.Module):
    def __init__(
        self,
        seq_len=8,
        latent_dim=4,
        hidden_dims=(64, 32),
        num_bands=2
    ):
        super().__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim

        h1, h2 = hidden_dims

        self.pos_enc = PositionalEncoding1D(num_bands)
        feat_dim = 1 + 2 * num_bands
        input_dim = 2 * seq_len * feat_dim

        # ---------------- ENCODER ----------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            ResidualBlock(h1),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU()
        )

        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)

        # ---------------- DECODER ----------------
        # ========================================================
        # Shared decoder backbone
        # ========================================================
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.SiLU(),
            nn.Linear(h2, h1),
            nn.SiLU(),
            ResidualBlock(h1)
        )
        # Profile profile
        self.decoder_profile = nn.Sequential(
            nn.Linear(h1, h2),
            nn.SiLU(),
            nn.Linear(h2, seq_len),
            nn.Sigmoid()
        )

        # Profile time coordinates
        self.decoder_t_profile = nn.Sequential(
            nn.Linear(h1, h2),
            nn.SiLU(),
            nn.Linear(h2, seq_len - 1)
        )



    def encode(self, x):
        # x: (batch, 2, seq_len)
        b, c, s = x.shape

        if c != 2 or s != self.seq_len:
            raise ValueError(
                f"Expected (batch, 2, {self.seq_len}), "
                f"received {tuple(x.shape)}"
            )

        x_flat = x.reshape(b * c, s)
        x_encoded = self.pos_enc(x_flat)
        h = x_encoded.reshape(b, -1)

        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu

        logvar = torch.clamp(logvar, -12, 8)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



    def decode(self, z):

        h = self.decoder(z)

        # ========================================================
        # Profile profile
        # ========================================================
        recon_profile = self.decoder_profile(h)

        # Positive time intervals for profile
        raw_dt = self.decoder_t_profile(h)
        dt = F.softplus(raw_dt) + 1e-5

        # Normalize total duration to 1
        dt = dt / dt.sum(dim=-1, keepdim=True)

        zeros = torch.zeros_like(dt[:, :1])

        recon_t = torch.cat(
            [zeros, torch.cumsum(dt, dim=-1)],
            dim=-1
        )

        return torch.stack([recon_t, recon_profile], dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def encode_profiles(profiles, seed=42, epochs=60):
    """Fit separate VAEs to the campaign profiles and return posterior means."""
    profiles = torch.as_tensor(np.asarray(profiles), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ImprovedTrajectoryVAE(seq_len=profiles.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        for epoch in range(epochs):
            for batch in torch.randperm(len(profiles), generator=generator).split(64):
                target = profiles[batch]
                reconstructed, mean, log_variance = model(target)
                kl = (-0.5 * (1 + log_variance - mean.square() - log_variance.exp())).mean()
                loss = (F.mse_loss(reconstructed, target)
                        + 0.3 * F.mse_loss(reconstructed.diff(), target.diff())
                        + min(1, (epoch + 1) / 20) * 0.001 * kl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            mean, _ = model.encode(profiles)
            return mean.numpy()
