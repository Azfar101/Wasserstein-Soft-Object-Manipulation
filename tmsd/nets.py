"""
Networks for TMSD: the external-state representation φ, and a standard
skill-conditioned SAC actor/critic pair.

φ maps the soil mass histogram (and nothing else) to a D-dim latent.
Keeping proprioception out of φ is a design invariant, not a detail:
it is what makes arm-only motion reward-free. The policy and critics,
by contrast, see the full observation plus the skill vector z.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class Phi(nn.Module):
    """Representation φ: soil histogram → ℝ^D (external state only)."""

    def __init__(self, hist_bins: int, skill_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(hist_bins, hidden, skill_dim)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return self.net(hist)


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian π(a | obs, z)."""

    def __init__(self, obs_dim: int, skill_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = mlp(obs_dim + skill_dim, hidden, 2 * act_dim)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor, z: torch.Tensor):
        mean, log_std = self.trunk(torch.cat([obs, z], dim=-1)).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, z: torch.Tensor):
        """Reparameterized action + log-prob (with tanh correction)."""
        mean, log_std = self(obs, z)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        a = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log1p(-a.pow(2) + 1e-6)
        return a, log_prob.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False):
        mean, log_std = self(obs, z)
        if deterministic:
            return torch.tanh(mean)
        return torch.tanh(torch.distributions.Normal(mean, log_std.exp()).sample())


class TwinQ(nn.Module):
    """Twin Q-networks Q(obs, z, a)."""

    def __init__(self, obs_dim: int, skill_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + skill_dim + act_dim
        self.q1 = mlp(in_dim, hidden, 1)
        self.q2 = mlp(in_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, a: torch.Tensor):
        x = torch.cat([obs, z, a], dim=-1)
        return self.q1(x), self.q2(x)


class SkillDynamics(nn.Module):
    """DADS-style skill dynamics q(Δx | x, z): predicts the state delta."""

    def __init__(self, state_dim: int, skill_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(state_dim + skill_dim, hidden, state_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=-1))


def sample_skills(batch: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples on the unit sphere S^{dim−1}."""
    v = rng.normal(size=(batch, dim))
    v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v.astype(np.float32)
