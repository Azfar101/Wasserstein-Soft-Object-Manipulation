"""
:class:`ExcavatorEnv` — a Gymnasium environment wrapping the DEM soil + arm sim.

The agent commands the three arm joints; the world is the deformable soil bed
(optionally with a boulder). One env step advances ``sub_steps × dt`` seconds of
physics. Observations are fixed-size (arm proprioception + bucket/payload state +
a soil heightfield + boulder summary). Reward/termination come from a pluggable
:class:`~excavator_sim.tasks.Task` (default: :class:`ExcavationTask`).

Example
-------
>>> from excavator_sim import ExcavatorEnv
>>> env = ExcavatorEnv(render_mode=None)
>>> obs, info = env.reset(seed=0)
>>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - gymnasium is a hard dependency
    raise ImportError("excavator_sim.env requires gymnasium; `pip install gymnasium`") from exc

from .config import SimConfig
from .particles import ParticleSystem, spawn_pile, place_boulder
from .solver import Solver, BucketState
from .arm import ArmDynamics
from . import geometry
from . import physics_feedback as pf
from .tasks import Task, ExcavationTask, RewardFnTask

# Observation normalisation scales (keep components ~O(1)).
_PAYLOAD_NORM = 500.0     # kg
_REACTION_NORM = 1.0e5    # N
_OBS_CLIP = 5.0
# Cartesian (position-mode) action → per-step nudge magnitudes.
_POS_XY_SPEED = 1.2       # m/s of wrist target travel at full command
_POS_ROT_SPEED = 1.5      # rad/s of bucket rotation at full command


class ExcavatorEnv(gym.Env):
    """Gymnasium env: drive an excavator arm to dig a 2D DEM soil bed."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config: Optional[SimConfig] = None, *,
                 render_mode: Optional[str] = None,
                 task: Optional[Task] = None,
                 reward_fn=None,
                 max_episode_steps: int = 2000,
                 force_cpu: bool = False):
        super().__init__()
        self.cfg = config or SimConfig()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._frame_dt = self.cfg.dt * self.cfg.sub_steps

        if task is not None:
            self.task = task
        elif reward_fn is not None:
            self.task = RewardFnTask(self.cfg.task, reward_fn)
        else:
            self.task = ExcavationTask(self.cfg.task)

        # Simulation components.
        self.ps = ParticleSystem(self.cfg.particle.max_particles)
        self.solver = Solver(self.cfg, force_cpu=force_cpu)
        self.arm = ArmDynamics(self.cfg)
        self._renderer = None

        # Per-step feedback cache (filled by reset/step, read by obs/reward/render).
        self.arm_state = self.arm.state()
        self.payload_mass = 0.0
        self.payload_cm = (0.0, 0.0)
        self.contact_torques = {"shoulder": 0.0, "elbow": 0.0, "wrist": 0.0}
        self.boulder = pf.boulder_info(self.ps, 0.0, 0.0, self.cfg.bucket.radius, 0,
                                       self.cfg.boulder.bury_threshold)
        self.steps = 0

        # Spaces.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        dim = 19 + self.cfg.heightfield_bins
        self.observation_space = spaces.Box(-_OBS_CLIP, _OBS_CLIP, shape=(dim,),
                                            dtype=np.float32)

    # ── Gymnasium API ───────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        self.ps.clear()
        spawn_pile(self.ps, self.cfg, self.np_random)

        boulder_xy = options.get("boulder_xy")
        if boulder_xy is None and options.get("boulder", False):
            bx = self.np_random.uniform(self.cfg.arm.pivot_x + 1.0, self.cfg.domain_width - 1.0)
            by = self.np_random.uniform(0.4, self.cfg.domain_height * self.cfg.spawn.pile_height_frac)
            boulder_xy = (bx, by)
        if boulder_xy is not None:
            place_boulder(self.ps, self.cfg, *boulder_xy)

        self.arm = ArmDynamics(self.cfg)
        self.steps = 0

        for _ in range(int(options.get("settle_steps", 0))):
            self.solver.step(self.ps, self.cfg)

        self._refresh()
        self.task.reset(self)
        return self._build_obs(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self._apply_action(action)
        st = self.arm.state()                       # pose driving collider + torque
        self.solver.step(self.ps, self.cfg, bucket=self._bucket_state(st))

        payload = pf.detect_payload(self.ps, st, self.cfg.bucket.radius)
        ct = pf.contact_torques(st, self.solver.bucket_reaction_fx,
                                self.solver.bucket_reaction_fy,
                                self.cfg.arm.pivot_x, self.cfg.arm.pivot_y)
        self.arm.set_payload(payload[0])
        self.arm.integrate(self._frame_dt, ct)

        self.steps += 1
        self._refresh(payload, ct)

        reward = float(self.task.reward(self, action))
        terminated = bool(self.task.terminated(self))
        truncated = self.steps >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return self._build_obs(), reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .renderer import Renderer
            self._renderer = Renderer(self.cfg, mode=self.render_mode)
        return self._renderer.draw(self.ps, self.arm_state,
                                   contact_torques=self.contact_torques,
                                   info=self._info())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── action / bucket plumbing ────────────────────────────────────
    def _apply_action(self, action):
        if self.cfg.control_mode == "velocity":
            self.arm.command_joint_velocity(action)
        else:  # position / Cartesian
            self.arm.nudge_wrist_target(action[0] * _POS_XY_SPEED * self._frame_dt,
                                        action[1] * _POS_XY_SPEED * self._frame_dt)
            self.arm.nudge_wrist_rotation(action[2] * _POS_ROT_SPEED * self._frame_dt)

    def _bucket_state(self, st) -> BucketState:
        segs = geometry.bucket_segments(
            st["wrist_x"], st["wrist_y"], st["mouth_ax"], st["mouth_ay"],
            st["stick_ax"], st["stick_ay"], self.cfg.bucket.radius)
        bvx, bvy = self.arm.wrist_linear_velocity()
        return BucketState(active=True, seg_x0=segs[0], seg_y0=segs[1],
                           seg_x1=segs[2], seg_y1=segs[3], vx=bvx, vy=bvy)

    # ── feedback cache ──────────────────────────────────────────────
    def _refresh(self, payload=None, ct=None):
        self.arm_state = self.arm.state()
        if payload is None:
            payload = pf.detect_payload(self.ps, self.arm_state, self.cfg.bucket.radius)
        self.payload_mass = payload[0]
        self.payload_cm = (payload[1], payload[2])
        if ct is not None:
            self.contact_torques = ct
        self.boulder = pf.boulder_info(
            self.ps, self.arm_state["wrist_x"], self.arm_state["wrist_y"],
            self.cfg.bucket.radius, self.solver.bucket_contact_count,
            self.cfg.boulder.bury_threshold)

    # ── observation ─────────────────────────────────────────────────
    def _heightfield(self) -> np.ndarray:
        n = self.ps.count
        k = self.cfg.heightfield_bins
        h = np.zeros(k, dtype=np.float64)
        soil = self.ps.soil_mask
        if soil.any():
            px = self.ps.px[:n][soil]
            py = self.ps.py[:n][soil]
            bins = np.clip((px / self.cfg.domain_width * k).astype(np.int64), 0, k - 1)
            np.maximum.at(h, bins, py)
        return h / self.cfg.domain_height

    def _build_obs(self) -> np.ndarray:
        st = self.arm_state
        cfg = self.cfg
        lim = cfg.arm.limits
        mo = cfg.arm.max_omega

        def na(a, lo, hi):
            return 2.0 * (a - lo) / (hi - lo) - 1.0

        mouth_angle = np.arctan2(st["mouth_ay"], st["mouth_ax"]) / np.pi
        parts = [
            na(st["shoulder_angle"], *lim[0]),
            na(st["elbow_angle"], *lim[1]),
            na(st["wrist_angle"], *lim[2]),
            st["omega_shoulder"] / mo[0],
            st["omega_elbow"] / mo[1],
            st["omega_wrist"] / mo[2],
            st["wrist_x"] / cfg.domain_width * 2.0 - 1.0,
            st["wrist_y"] / cfg.domain_height * 2.0 - 1.0,
            mouth_angle,
            st["load_shoulder"], st["load_elbow"], st["load_wrist"],
            self.payload_mass / _PAYLOAD_NORM,
            1.0 if self.solver.bucket_contact_count > 0 else 0.0,
            self.solver.bucket_reaction_fx / _REACTION_NORM,
            self.solver.bucket_reaction_fy / _REACTION_NORM,
        ]
        parts.extend(self._heightfield().tolist())
        b = self.boulder
        parts.extend([
            b["rel_x"] / cfg.domain_width if b["present"] else 0.0,
            b["rel_y"] / cfg.domain_height if b["present"] else 0.0,
            1.0 if b["present"] and b["buried"] else 0.0,
        ])
        obs = np.asarray(parts, dtype=np.float32)
        return np.clip(obs, -_OBS_CLIP, _OBS_CLIP)

    def _info(self) -> dict:
        info = {
            "payload_mass": self.payload_mass,
            "contact_torques": dict(self.contact_torques),
            "contact_count": self.solver.bucket_contact_count,
            "bucket_reaction": (self.solver.bucket_reaction_fx, self.solver.bucket_reaction_fy),
            "wrist_xy": (self.arm_state["wrist_x"], self.arm_state["wrist_y"]),
            "boulder": dict(self.boulder),
            "backend": "cuda" if self.solver.use_cuda else "cpu",
        }
        info.update(self.task.info(self))
        return info
