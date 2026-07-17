"""
Goal-conditioned SAC for direct soil-shaping control.

The unsupervised skill vocabulary proved too coarse for arbitrary user
goals (MPC with ground-truth lookahead barely improved on greedy — a
vocabulary limit, not a selection limit). This trainer attacks the demo
task head-on: π(a | obs, goal_profile) with a dense physical reward

    r_t = ( W₂(h_t, g) − W₂(h_{t+1}, g) ) · scale

i.e. per-step transport progress toward the goal, measured by the same
closed-form 1D W₂ used everywhere else. Rewards are recomputed on GPU
at update time, so hindsight relabeling is free: every episode is
pushed twice — once with its commanded goal, once with its achieved
terminal state as the goal (final-strategy HER).

This is the deployable control layer; the skill-discovery results
remain the paper's science.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .metrics import w2_1d
from .nets import GaussianPolicy, TwinQ


@dataclass
class GCConfig:
    obs_dim: int = 35
    goal_dim: int = 64          # goal = 64-bin ground mass profile
    act_dim: int = 3
    reward_scale: float = 100.0
    n_quantiles: int = 2048

    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    hidden: int = 256
    target_entropy: float | None = None
    buffer_capacity: int = 400_000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class GCTrainer:
    """SAC on (obs, goal) with batch-recomputed W₂-progress rewards."""

    def __init__(self, cfg: GCConfig, grid: np.ndarray):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.grid = torch.as_tensor(grid, dtype=torch.float32, device=self.device)

        self.policy = GaussianPolicy(cfg.obs_dim, cfg.goal_dim, cfg.act_dim,
                                     cfg.hidden).to(self.device)
        self.q = TwinQ(cfg.obs_dim, cfg.goal_dim, cfg.act_dim,
                       cfg.hidden).to(self.device)
        self.q_target = TwinQ(cfg.obs_dim, cfg.goal_dim, cfg.act_dim,
                              cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.target_entropy = (cfg.target_entropy if cfg.target_entropy is not None
                               else -float(cfg.act_dim))
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

        # Reuse ReplayBuffer: the "skill" slot stores the goal profile.
        self.buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim,
                                   cfg.goal_dim, cfg.act_dim, cfg.goal_dim,
                                   self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray,
            deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        g_t = torch.as_tensor(goal, device=self.device).unsqueeze(0)
        return self.policy.act(obs_t, g_t, deterministic).squeeze(0).cpu().numpy()

    def push_episode(self, transitions, goal: np.ndarray) -> None:
        """Store an episode twice: commanded goal + terminal-state HER."""
        if not transitions:
            return
        terminal_hist = transitions[-1][4]  # next_hist of the last step
        for relabeled in (goal, terminal_hist):
            for (obs, hist, act, next_obs, next_hist) in transitions:
                self.buffer.push(obs, hist, act, next_obs, next_hist,
                                 relabeled, False)

    def update(self) -> dict:
        cfg = self.cfg
        b = self.buffer.sample(cfg.batch_size, self.rng)
        g = b["skill"]  # goal profiles live in the skill slot

        with torch.no_grad():
            r = (w2_1d(b["hist"], g, self.grid, n_quantiles=cfg.n_quantiles)
                 - w2_1d(b["next_hist"], g, self.grid,
                         n_quantiles=cfg.n_quantiles))
            r = (r * cfg.reward_scale).unsqueeze(-1)

        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.policy.sample(b["next_obs"], g)
            q1_t, q2_t = self.q_target(b["next_obs"], g, a2)
            target = r + cfg.gamma * (torch.min(q1_t, q2_t) - alpha * logp2)
        q1, q2 = self.q(b["obs"], g, b["act"])
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_q.step()

        a, logp = self.policy.sample(b["obs"], g)
        q1_pi, q2_pi = self.q(b["obs"], g, a)
        actor_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        self.opt_policy.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_policy.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.lerp_(p, cfg.tau)

        return {"gc/reward_mean": r.mean().item(),
                "gc/q_loss": q_loss.item(),
                "gc/actor_loss": actor_loss.item(),
                "gc/alpha": alpha.item()}

    def save(self, path: str) -> None:
        torch.save({"cfg": self.cfg, "kind": "gc",
                    "policy": self.policy.state_dict(),
                    "q": self.q.state_dict(),
                    "q_target": self.q_target.state_dict(),
                    "log_alpha": self.log_alpha.detach().cpu()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
