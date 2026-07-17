"""
Skill-MPC: model-predictive control over the learned skill vocabulary,
using the simulator itself as the (perfect) dynamics model.

Zero-shot latent steering (z = φ(goal) − φ(now)) trusts φ's geometry,
which collapses on states outside the training distribution — exactly
the states an interactive user creates. This planner removes that trust:
snapshot the sim, roll each candidate skill forward for a horizon,
score the *actual* resulting W₂ to the goal, restore the snapshot,
execute the winner for real. The skills remain frozen; planning is
pure search over which one to fire.

Cost: K × horizon simulated steps per decision (a few seconds of
wall-clock "thinking"). Honesty: if no candidate improves on doing
nothing, the planner says so instead of thrashing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .metrics import w2_1d_exact


@dataclass
class SimSnapshot:
    particles: dict
    count: int
    arm: object
    steps: int


def snapshot(env) -> SimSnapshot:
    """Capture the full mutable sim state of an ExcavatorEnv."""
    e = env.env
    ps = e.ps
    return SimSnapshot(
        particles={name: getattr(ps, name)[:ps.count].copy()
                   for name in ps._ARRAYS},
        count=ps.count,
        arm=copy.deepcopy(e.arm),
        steps=e.steps,
    )


def restore(env, snap: SimSnapshot) -> np.ndarray:
    """Write a snapshot back; returns the rebuilt observation."""
    e = env.env
    ps = e.ps
    ps.count = snap.count
    for name, arr in snap.particles.items():
        getattr(ps, name)[:snap.count] = arr
    e.arm = copy.deepcopy(snap.arm)
    e.steps = snap.steps
    e._refresh()
    return e._build_obs()


class SkillMPC:
    """Greedy one-skill lookahead with ground-truth rollouts."""

    def __init__(self, env, trainer, *, n_candidates: int = 10,
                 horizon: int = 80, seed: int = 0):
        self.env = env
        self.trainer = trainer
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        dim = trainer.cfg.skill_dim
        # Fixed antipodal axis directions + random fill — cheap coverage
        # of the skill sphere that stays identical across decisions.
        cands = []
        for i in range(dim):
            v = np.zeros(dim, dtype=np.float32)
            v[i] = 1.0
            cands.append(v.copy())
            v[i] = -1.0
            cands.append(v.copy())
        while len(cands) < n_candidates:
            v = self.rng.normal(size=dim).astype(np.float32)
            cands.append(v / (np.linalg.norm(v) + 1e-12))
        self.candidates = np.stack(cands[:max(n_candidates, 2 * dim)])

    def _rollout_score(self, obs, z, target_hist) -> float:
        env = self.env
        for _ in range(self.horizon):
            obs, _, term, trunc, _ = env.step(
                self.trainer.act(obs, z, deterministic=True))
            if term or trunc:
                break
        return w2_1d_exact(env.eval_hist_1d(), target_hist, env.grid_eval)

    def plan(self, obs, target_hist, on_progress=None):
        """Try every candidate skill in imagination; return
        (best_z or None, predicted_w2, baseline_w2). None = nothing helps."""
        base = snapshot(self.env)
        baseline = w2_1d_exact(self.env.eval_hist_1d(), target_hist,
                               self.env.grid_eval)
        best_z, best_w2 = None, baseline - 1e-3  # must beat doing nothing
        for i, z in enumerate(self.candidates):
            score = self._rollout_score(obs, z, target_hist)
            obs = restore(self.env, base)
            if score < best_w2:
                best_z, best_w2 = z.copy(), score
            if on_progress is not None:
                on_progress(i + 1, len(self.candidates), best_w2)
        return best_z, best_w2, baseline
