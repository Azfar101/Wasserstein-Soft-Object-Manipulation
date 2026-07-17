"""
TMSD trainer: skill-conditioned SAC + METRA-style constrained
representation learning, with a physical transport ground metric.

Objective (METRA template, generalized ground metric):

    max_{π, φ}  E[(φ(h′) − φ(h)) · z]
    s.t.        ‖φ(h) − φ(h′)‖ ≤ d(h, h′)   for adjacent (h, h′)

where h is the soil mass histogram and d is a pluggable ground metric
(default: closed-form 1D W₂, i.e. physical transport cost in meters).
The constraint is enforced with a dual variable λ (log-parametrized)
on the penalty min(slack, d² − ‖Δφ‖²); the slack is *relative* to d²
because W₂ steps span orders of magnitude, unlike METRA's constant
temporal distance of 1.

The SAC intrinsic reward r = (φ(h′) − φ(h)) · z is recomputed from the
current φ at every gradient step (the reward is nonstationary by
design; replaying stale rewards would anchor the policy to old φ).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .metrics import GROUND_METRICS, SlicedW2
from .nets import Phi, GaussianPolicy, TwinQ, SkillDynamics


@dataclass
class TMSDConfig:
    # Dimensions
    obs_dim: int = 35
    hist_bins: int = 64
    act_dim: int = 3
    skill_dim: int = 2

    # Objective family: "metra" = constrained distance-maximizing (default);
    # "mi" = continuous-DIAYN mutual information — discriminator d(x) predicts
    # the episode skill, reward = cos(d(x′), z). No ground metric, no dual.
    # "dads" = skill-dynamics predictability: q(Δx|x,z) trained by regression,
    # reward = log q under own z minus log mean q under L prior samples.
    objective: str = "metra"
    dads_sigma: float = 0.01          # Gaussian scale for DADS log-densities
    dads_alt_samples: int = 16        # L alternative skills for the marginal

    # Ground metric
    metric: str = "w2"                # key into GROUND_METRICS
    n_quantiles: int = 2048           # W₂ quadrature resolution
    n_projections: int = 64           # sliced-W₂ only (2D measures)
    # What φ sees: "hist" = external soil state only (TMSD invariant);
    # "obs" = full observation incl. proprioception (vanilla-METRA-style
    # baseline for the external-conditioning ablation).
    phi_input: str = "hist"
    # Measure the wrapper produced ("1d" x-marginal | "2d" grid); recorded
    # so eval scripts can rebuild a matching env from the checkpoint.
    measure: str = "1d"
    # W₂ steps are ~1e-3 m, so raw rewards would drown in SAC's entropy
    # term; scaling only the reward (not the constraint) fixes conditioning
    # without changing the optimal φ.
    reward_scale: float = 100.0

    # Constraint dual
    dual_lam_init: float = 30.0
    dual_slack_rel: float = 0.25      # cap penalty at rel·d² (+ tiny abs floor)
    dual_slack_abs: float = 1e-8
    lr_dual: float = 1e-3

    # SAC
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    hidden: int = 256
    target_entropy: float | None = None   # default: −act_dim
    buffer_capacity: int = 300_000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class TMSDTrainer:
    def __init__(self, cfg: TMSDConfig, grid: np.ndarray):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.grid = torch.as_tensor(grid, dtype=torch.float32, device=self.device)
        if cfg.metric == "sliced_w2":
            self.metric_fn = SlicedW2(self.grid,
                                      getattr(cfg, "n_projections", 64))
        else:
            self.metric_fn = GROUND_METRICS[cfg.metric]

        self.phi_input = getattr(cfg, "phi_input", "hist")  # getattr: old ckpts
        phi_in_dim = cfg.obs_dim if self.phi_input == "obs" else cfg.hist_bins
        self.phi = Phi(phi_in_dim, cfg.skill_dim, cfg.hidden).to(self.device)
        self.dyn = None
        if getattr(cfg, "objective", "metra") == "dads":
            self.dyn = SkillDynamics(phi_in_dim, cfg.skill_dim,
                                     cfg.hidden).to(self.device)
            self.opt_dyn = torch.optim.Adam(self.dyn.parameters(), lr=cfg.lr)
        self.policy = GaussianPolicy(cfg.obs_dim, cfg.skill_dim, cfg.act_dim,
                                     cfg.hidden).to(self.device)
        self.q = TwinQ(cfg.obs_dim, cfg.skill_dim, cfg.act_dim, cfg.hidden).to(self.device)
        self.q_target = TwinQ(cfg.obs_dim, cfg.skill_dim, cfg.act_dim,
                              cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.log_lam = torch.tensor(float(np.log(cfg.dual_lam_init)),
                                    device=self.device, requires_grad=True)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.target_entropy = (cfg.target_entropy if cfg.target_entropy is not None
                               else -float(cfg.act_dim))

        self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=cfg.lr)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.opt_lam = torch.optim.Adam([self.log_lam], lr=cfg.lr_dual)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

        self.buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim, cfg.hist_bins,
                                   cfg.act_dim, cfg.skill_dim, self.device)

    # ── acting ──────────────────────────────────────────────────────
    def sample_skill(self) -> np.ndarray:
        v = self.rng.normal(size=self.cfg.skill_dim)
        return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, z: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        z_t = torch.as_tensor(z, device=self.device).unsqueeze(0)
        a = self.policy.act(obs_t, z_t, deterministic)
        return a.squeeze(0).cpu().numpy()

    # ── intrinsic reward (current φ, no grad) ───────────────────────
    @torch.no_grad()
    def intrinsic_reward(self, phi_in: torch.Tensor, next_phi_in: torch.Tensor,
                         z: torch.Tensor) -> torch.Tensor:
        """phi_in / next_phi_in must match cfg.phi_input (hist or full obs)."""
        objective = getattr(self.cfg, "objective", "metra")
        if objective == "mi":
            pred = F.normalize(self.phi(next_phi_in), dim=-1)
            return (pred * z).sum(dim=-1, keepdim=True) * self.cfg.reward_scale
        if objective == "dads":
            return self._dads_reward(phi_in, next_phi_in, z)
        dphi = self.phi(next_phi_in) - self.phi(phi_in)
        return (dphi * z).sum(dim=-1, keepdim=True) * self.cfg.reward_scale

    @torch.no_grad()
    def _dads_reward(self, x: torch.Tensor, x_next: torch.Tensor,
                     z: torch.Tensor) -> torch.Tensor:
        """DADS diversity reward: log q(Δx|x,z) − log(1/L Σ_l q(Δx|x,z_l))."""
        cfg = self.cfg
        B = x.shape[0]
        L = cfg.dads_alt_samples
        inv2s2 = 1.0 / (2.0 * cfg.dads_sigma ** 2)
        delta = x_next - x
        err_z = ((self.dyn(x, z) - delta) ** 2).mean(dim=-1)           # (B,)
        z_alt = torch.randn(L, B, z.shape[-1], device=x.device)
        z_alt = z_alt / (z_alt.norm(dim=-1, keepdim=True) + 1e-12)
        x_rep = x.unsqueeze(0).expand(L, -1, -1)
        err_alt = ((self.dyn(x_rep.reshape(L * B, -1),
                             z_alt.reshape(L * B, -1)).reshape(L, B, -1)
                    - delta.unsqueeze(0)) ** 2).mean(dim=-1)           # (L, B)
        logp_z = -err_z * inv2s2
        log_marg = torch.logsumexp(-err_alt * inv2s2, dim=0) - np.log(L)
        r = (logp_z - log_marg).unsqueeze(-1) * cfg.reward_scale
        return r.clamp(-50.0, 50.0)

    # ── one gradient step ───────────────────────────────────────────
    def update(self) -> dict:
        cfg = self.cfg
        b = self.buffer.sample(cfg.batch_size, self.rng)
        stats = {}

        pin, pin_next = ((b["obs"], b["next_obs"]) if self.phi_input == "obs"
                         else (b["hist"], b["next_hist"]))

        objective = getattr(cfg, "objective", "metra")
        if objective == "mi":
            # 1. (MI) discriminator: predict the episode skill from the state.
            pred = F.normalize(self.phi(pin_next), dim=-1)       # (B, D)
            obj = (pred * b["skill"]).sum(dim=-1)                # cosine
            phi_loss = -obj.mean()
            self.opt_phi.zero_grad(set_to_none=True)
            phi_loss.backward()
            self.opt_phi.step()
            stats["phi/obj"] = obj.mean().item()
        elif objective == "dads":
            # 1. (DADS) skill-dynamics regression on observed transitions.
            delta = pin_next - pin
            dyn_loss = F.mse_loss(self.dyn(pin, b["skill"]), delta)
            self.opt_dyn.zero_grad(set_to_none=True)
            dyn_loss.backward()
            self.opt_dyn.step()
            stats["phi/obj"] = -dyn_loss.item()  # model fit (higher = better)
        else:
            # 1. (METRA) φ + dual λ on the transport-metric constraint.
            d = self.metric_fn(b["hist"], b["next_hist"], self.grid,
                               n_quantiles=cfg.n_quantiles)      # (B,)
            d_sq = d.pow(2)
            dphi = self.phi(pin_next) - self.phi(pin)            # (B, D)
            obj = (dphi * b["skill"]).sum(dim=-1)                # (B,)
            cst_raw = d_sq - dphi.pow(2).sum(dim=-1)             # ≥0 ⇔ satisfied
            slack = cfg.dual_slack_rel * d_sq + cfg.dual_slack_abs
            cst = torch.minimum(cst_raw, slack)
            lam = self.log_lam.exp().detach()
            phi_loss = -(obj + lam * cst).mean()
            self.opt_phi.zero_grad(set_to_none=True)
            phi_loss.backward()
            self.opt_phi.step()

            lam_loss = self.log_lam * cst.mean().detach()
            self.opt_lam.zero_grad(set_to_none=True)
            lam_loss.backward()
            self.opt_lam.step()

            stats["phi/obj"] = obj.mean().item()
            stats["phi/constraint"] = cst_raw.mean().item()
            stats["phi/violation_rate"] = (cst_raw < 0).float().mean().item()
            stats["phi/lambda"] = lam.item()
            stats["metric/d_mean"] = d.mean().item()
            stats["metric/d_nonzero"] = (d > 0).float().mean().item()

        # 2. Critic on the recomputed intrinsic reward.
        r = self.intrinsic_reward(pin, pin_next, b["skill"])
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.policy.sample(b["next_obs"], b["skill"])
            q1_t, q2_t = self.q_target(b["next_obs"], b["skill"], a2)
            q_next = torch.min(q1_t, q2_t) - alpha * logp2
            target = r + cfg.gamma * (1.0 - b["done"]) * q_next
        q1, q2 = self.q(b["obs"], b["skill"], b["act"])
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_q.step()

        # 3. Actor + temperature.
        a, logp = self.policy.sample(b["obs"], b["skill"])
        q1_pi, q2_pi = self.q(b["obs"], b["skill"], a)
        actor_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        self.opt_policy.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_policy.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        # 4. Polyak target update.
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.lerp_(p, cfg.tau)

        stats["sac/reward_mean"] = r.mean().item()
        stats["sac/q_loss"] = q_loss.item()
        stats["sac/actor_loss"] = actor_loss.item()
        stats["sac/alpha"] = alpha.item()
        return stats

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: str) -> None:
        payload = {
            "cfg": self.cfg,
            "phi": self.phi.state_dict(),
            "policy": self.policy.state_dict(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "log_lam": self.log_lam.detach().cpu(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.dyn is not None:
            payload["dyn"] = self.dyn.state_dict()
        torch.save(payload, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.phi.load_state_dict(ckpt["phi"])
        self.policy.load_state_dict(ckpt["policy"])
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        if self.dyn is not None and "dyn" in ckpt:
            self.dyn.load_state_dict(ckpt["dyn"])
        with torch.no_grad():
            self.log_lam.copy_(ckpt["log_lam"].to(self.device))
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
