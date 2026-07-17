"""
Replay buffer for TMSD.

Stores transitions *without* rewards: the intrinsic reward
r = (φ(h′) − φ(h))·z is recomputed with the current φ at sampling time
(as METRA does), so the buffer holds the soil histograms h, h′ needed
to evaluate both the reward and the W₂ ground metric per batch.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, hist_bins: int,
                 act_dim: int, skill_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.full = False
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.hist = np.zeros((capacity, hist_bins), dtype=np.float32)
        self.next_hist = np.zeros((capacity, hist_bins), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.skill = np.zeros((capacity, skill_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def push(self, obs, hist, act, next_obs, next_hist, skill, done: bool):
        i = self.idx
        self.obs[i] = obs
        self.hist[i] = hist
        self.act[i] = act
        self.next_obs[i] = next_obs
        self.next_hist[i] = next_hist
        self.skill[i] = skill
        self.done[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        n = len(self)
        idx = rng.integers(0, n, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)
        return {
            "obs": to(self.obs), "next_obs": to(self.next_obs),
            "hist": to(self.hist), "next_hist": to(self.next_hist),
            "act": to(self.act), "skill": to(self.skill), "done": to(self.done),
        }
