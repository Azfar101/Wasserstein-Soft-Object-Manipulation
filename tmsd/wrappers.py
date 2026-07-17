"""
Reward-free skill-discovery wrapper around :class:`ExcavatorEnv`.

Uses the base :class:`~excavator_sim.tasks.Task` (zero reward, never
terminates), disables boulders, shortens episodes, and computes the
soil mass histogram every step (returned in ``info["soil_hist"]``).
The wrapper is where "external state" is defined — everything TMSD
knows about the world outside the robot flows through here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from excavator_sim import ExcavatorEnv
from excavator_sim.config import SimConfig
from excavator_sim.tasks import Task

from excavator_sim.arm import ArmDynamics

from .measures import (soil_mass_histogram, bin_centers,
                       soil_mass_histogram_2d, bin_centers_2d,
                       soil_state_composite)


class SkillDiscoveryEnv:
    """Thin, reward-free wrapper. Not a gym.Wrapper on purpose — the
    interface is exactly what the trainer needs and nothing more."""

    def __init__(self, config: Optional[SimConfig] = None, *,
                 measure: str = "1d",  # "1d" x-marginal | "2d" grid | "composite"
                 hist_bins: int = 64,          # 1d/composite ground resolution
                 bins_x: int = 48, bins_y: int = 24,   # 2d measure resolution
                 air_bins: int = 16,           # composite airborne resolution
                 max_episode_steps: int = 200,
                 settle_steps: int = 30,
                 randomize_terrain: bool = False,
                 brush_ops: int = 0,           # random sculpt edits at reset
                 persist_soil_prob: float = 0.0,  # keep soil across episodes
                 arm_random_steps: int = 0,    # random actions to vary arm pose
                 render_mode: Optional[str] = None,
                 force_cpu: bool = False):
        if randomize_terrain and config is None:
            config = SimConfig()
            # Spawn a taller bed so the sculpting envelope has room to
            # carve mounds and valleys out of it.
            config.spawn.target_count = 1600
        self.env = ExcavatorEnv(config, render_mode=render_mode, task=Task(),
                                max_episode_steps=max_episode_steps,
                                force_cpu=force_cpu)
        self.measure = measure
        self.hist_bins = hist_bins
        self.bins_x, self.bins_y = bins_x, bins_y
        self.air_bins = air_bins
        self.settle_steps = settle_steps
        self.randomize_terrain = randomize_terrain
        self.brush_ops = brush_ops
        self.persist_soil_prob = persist_soil_prob
        self.arm_random_steps = arm_random_steps
        if measure == "2d":
            self.grid = bin_centers_2d(self.env.cfg, bins_x, bins_y)  # (K, 2)
            self.measure_dim = bins_x * bins_y
        elif measure == "composite":
            self.measure_dim = hist_bins + 3 + air_bins
            # Heterogeneous channels: no transport grid; use euclidean/temporal.
            self.grid = np.zeros(self.measure_dim)
        else:
            self.grid = bin_centers(self.env.cfg, hist_bins)          # (K,)
            self.measure_dim = hist_bins
        # 1D x-marginal on a fixed 64-bin grid: the common physical
        # yardstick for evaluation, regardless of what φ consumes.
        self.grid_eval = bin_centers(self.env.cfg, 64)
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))

    def _hist(self) -> np.ndarray:
        if self.measure == "2d":
            return soil_mass_histogram_2d(self.env.ps, self.env.cfg,
                                          self.bins_x, self.bins_y)
        if self.measure == "composite":
            return soil_state_composite(self.env.ps, self.env.cfg,
                                        self.env.arm_state,
                                        self.env.cfg.bucket.radius,
                                        self.hist_bins, self.air_bins)
        return soil_mass_histogram(self.env.ps, self.env.cfg, self.hist_bins)

    def measure_target_from_hist(self, hist: np.ndarray) -> np.ndarray:
        """Lift a 1D ground-profile goal into the measure space (goal = all
        mass on the ground in that profile, bucket empty, nothing in flight)."""
        if self.measure == "composite":
            return np.concatenate([hist.astype(np.float32),
                                   np.zeros(3 + self.air_bins, np.float32)])
        if self.measure == "1d":
            return hist.astype(np.float32)
        raise ValueError(f"no hist->target lift for measure {self.measure!r}")

    def eval_hist_1d(self) -> np.ndarray:
        """1D x-marginal (64 bins) for cross-run evaluation metrics."""
        return soil_mass_histogram(self.env.ps, self.env.cfg, 64)

    def hist_to_heights(self, hist: np.ndarray) -> np.ndarray:
        """Convert a mass histogram to visual surface heights (m), calibrated
        against the CURRENT soil state so packing/density cancel out.

        Used to draw goal-profile overlays in the renderer."""
        ps, cfg = self.env.ps, self.env.cfg
        k = len(hist)
        n = ps.count
        soil = ps.soil_mask
        heights_now = np.zeros(k)
        px = ps.px[:n][soil]
        py = ps.py[:n][soil]
        bins = np.clip((px / cfg.domain_width * k).astype(np.int64), 0, k - 1)
        np.maximum.at(heights_now, bins, py)
        hist_now = soil_mass_histogram(ps, cfg, k)
        ok = hist_now > 1e-6
        scale = np.median(heights_now[ok] / hist_now[ok]) if ok.any() else 0.0
        return np.clip(hist * scale, 0.05, cfg.domain_height)

    def _sculpt_terrain(self) -> None:
        """Carve the flat spawned bed down to a random smooth height
        envelope h(x) = base + Σ aₖ cos(2πkx/W + ϕₖ), so every episode
        starts from a different soil configuration."""
        env, cfg = self.env, self.env.cfg
        rng = env.np_random
        W = cfg.domain_width
        n = env.ps.count
        px, py = env.ps.px[:n], env.ps.py[:n]
        # Scale the envelope to the actual spawned bed height so the cut
        # always bites, regardless of particle count / domain size.
        bed_top = float(np.quantile(py, 0.98))
        base = rng.uniform(0.55, 0.90) * bed_top
        envelope = np.full(n, base)
        for k in range(1, rng.integers(2, 5)):
            amp = rng.uniform(0.0, 0.25) * bed_top
            phase = rng.uniform(0.0, 2.0 * np.pi)
            envelope += amp * np.cos(2.0 * np.pi * k * px / W + phase)
        envelope = np.clip(envelope, 0.30 * bed_top, 1.05 * bed_top)
        env.ps.keep_where(py <= envelope)

    def _brush_edits(self) -> None:
        """Random push / remove / add edits — the same operations a user
        makes in the interactive demo, so training covers that state family."""
        env, cfg = self.env, self.env.cfg
        rng = env.np_random
        ps = env.ps
        for _ in range(self.brush_ops):
            op = rng.choice(["push", "remove", "add"])
            wx = rng.uniform(1.0, cfg.domain_width - 1.0)
            n = ps.count
            if op == "push" and n:
                wy = rng.uniform(0.3, cfg.domain_height * 0.5)
                dx = ps.px[:n] - wx
                dy = ps.py[:n] - wy
                hit = (dx * dx + dy * dy < 0.8 ** 2) & ps.soil_mask
                d = np.sqrt(dx[hit] ** 2 + dy[hit] ** 2) + 1e-6
                ps.vx[:n][hit] += 5.0 * dx[hit] / d
                ps.vy[:n][hit] += 2.5 * np.abs(dy[hit]) / d
            elif op == "remove" and n:
                wy = rng.uniform(0.3, cfg.domain_height * 0.4)
                ps.remove_near(wx, wy, rng.uniform(0.3, 0.7))
            elif op == "add":
                pp = cfg.particle
                for _ in range(int(rng.integers(10, 40))):
                    r = rng.uniform(pp.radius_min, pp.radius_max)
                    ps.add(float(np.clip(wx + rng.uniform(-0.4, 0.4),
                                         r, cfg.domain_width - r)),
                           float(rng.uniform(1.0, cfg.domain_height * 0.55)),
                           r, density=pp.density, friction=cfg.soil.friction)
            for _ in range(8):
                env.solver.step(ps, cfg)

    def apply_user_mess(self, n_ops: int, settle: int = 40) -> None:
        """Public hook: apply n user-like brush edits + settle. Used by the
        goal-conditioned trainer to construct restore tasks (snapshot goal,
        mess the soil, train to restore)."""
        saved = self.brush_ops
        self.brush_ops = n_ops
        self._brush_edits()
        self.brush_ops = saved
        for _ in range(settle):
            self.env.solver.step(self.env.ps, self.env.cfg)
        self.env._refresh()

    def _finish_reset(self):
        """Settle, randomize arm pose, refresh caches, build outputs."""
        env = self.env
        for _ in range(self.settle_steps):
            env.solver.step(env.ps, env.cfg)
        env._refresh()
        for _ in range(self.arm_random_steps):
            a = env.np_random.uniform(-1.0, 1.0, 3).astype(np.float32)
            env.step(a)
        env.steps = 0
        env._refresh()
        return env._build_obs(), self._hist(), env._info()

    def reset(self, seed: Optional[int] = None):
        env = self.env
        # Soil persistence: with some probability keep the current (possibly
        # skill-modified) soil and only re-home the arm — episodes then start
        # from realistic mid-manipulation states, not always fresh terrain.
        persist = (self.persist_soil_prob > 0.0 and env.ps.count > 300
                   and seed is None
                   and env.np_random.random() < self.persist_soil_prob)
        if persist:
            env.arm = ArmDynamics(env.cfg)
            env.steps = 0
            out = self._finish_reset()
            env.task.reset(env)
            return out

        if not self.randomize_terrain:
            obs, info = self.env.reset(seed=seed,
                                       options={"settle_steps": self.settle_steps})
            return obs, self._hist(), info
        # Randomized terrain: spawn, sculpt, brush-edit, settle.
        obs, info = self.env.reset(seed=seed, options={"settle_steps": 0})
        self._sculpt_terrain()
        if self.brush_ops:
            for _ in range(10):
                env.solver.step(env.ps, env.cfg)
            self._brush_edits()
        return self._finish_reset()

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = self.env.step(action)
        hist = self._hist()
        return obs, hist, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
