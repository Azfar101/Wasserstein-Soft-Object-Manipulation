"""
Velocity-governed hydraulic excavator-arm dynamics.

The arm is modelled the way a real hydraulic machine behaves, *not* as an
idealised torque-actuated linkage:

    operator/policy command → valve flow → joint angular velocity,
    soil contact load → reduced effective flow → the joint slows or stalls.

So a command sets a *desired joint velocity*; that desired velocity is throttled
by a contact-load factor and rate-limited by the hydraulic spool response, then
integrated. There is no torque integration or PID loop (an earlier version had
one; it oscillated during soil contact and was removed).

Two command sources are supported via ``control_mode``:

* ``"velocity"`` — a normalised ``[boom, stick, bucket]`` velocity command
  (natural for RL); see :meth:`command_joint_velocity`.
* ``"position"`` — joint targets driven from a Cartesian wrist target through
  IK (natural for a human operator / scripted motion); see
  :meth:`nudge_wrist_target`.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .config import SimConfig
from . import geometry

# Asymmetric load filter: register contact fast, release slowly (anti-chatter).
_ALPHA_RISE = 0.5
_ALPHA_FALL = 0.05


def _clamp(value, lo, hi):
    return lo if value < lo else (hi if value > hi else value)


class ArmDynamics:
    """Stateful arm: joint angles/velocities driven by the hydraulic model."""

    def __init__(self, config: SimConfig):
        a = config.arm
        self._pivot_x = a.pivot_x
        self._pivot_y = a.pivot_y
        self._boom_l = a.boom_length
        self._stick_l = a.stick_length
        self._limits = a.limits
        self._max_omega = list(a.max_omega)
        self._accel = list(a.accel)
        self._gain = a.velocity_gain
        self._payload_droop = a.payload_speed_droop
        # Runtime-adjustable stall torque (the demo exposes the shoulder value).
        self.stall_torque: List[float] = list(a.stall_torque)

        # The bucket collider may not pass below the domain floor (a solid wall).
        self._bucket_r = config.bucket.radius
        self._floor_y = 0.0

        self.control_mode = config.control_mode

        # Joint state (shoulder, elbow, wrist).
        self.shoulder_angle = math.pi * 0.1
        self.elbow_angle = math.pi * 0.8
        self.wrist_angle = 0.0
        self.omega = [0.0, 0.0, 0.0]

        # Position-mode targets.
        self.target_shoulder = self.shoulder_angle
        self.target_elbow = self.elbow_angle
        self.target_wrist = self.wrist_angle
        wx, wy, *_ = self._fk()
        self.target_wrist_x = wx
        self.target_wrist_y = wy

        # Velocity-mode command (normalised, set per step).
        self.cmd_omega = [0.0, 0.0, 0.0]

        self._filtered_load = [0.0, 0.0, 0.0]
        self._payload_mass = 0.0

    # ── kinematics ──────────────────────────────────────────────────
    def _fk(self):
        return geometry.forward_kinematics(
            self.shoulder_angle, self.elbow_angle, self.wrist_angle,
            self._pivot_x, self._pivot_y, self._boom_l, self._stick_l)

    def state(self) -> Dict[str, float]:
        """Full kinematic + load snapshot (used by env, renderer, HUD)."""
        (wx, wy, ex, ey, sax, say, max_, may) = self._fk()
        return {
            "shoulder_angle": self.shoulder_angle,
            "elbow_angle": self.elbow_angle,
            "wrist_angle": self.wrist_angle,
            "omega_shoulder": self.omega[0],
            "omega_elbow": self.omega[1],
            "omega_wrist": self.omega[2],
            "wrist_x": wx, "wrist_y": wy,
            "elbow_x": ex, "elbow_y": ey,
            "stick_ax": sax, "stick_ay": say,
            "mouth_ax": max_, "mouth_ay": may,
            "load_shoulder": self._filtered_load[0],
            "load_elbow": self._filtered_load[1],
            "load_wrist": self._filtered_load[2],
            "payload_mass": self._payload_mass,
        }

    def wrist_linear_velocity(self):
        """Wrist linear velocity (m/s) from joint rates via the Jacobian."""
        sa = self.shoulder_angle
        stick_abs = sa - (math.pi - self.elbow_angle)
        vex = -self._boom_l * math.sin(sa) * self.omega[0]
        vey = self._boom_l * math.cos(sa) * self.omega[0]
        omega_stick = self.omega[0] + self.omega[1]
        vwx = vex - self._stick_l * math.sin(stick_abs) * omega_stick
        vwy = vey + self._stick_l * math.cos(stick_abs) * omega_stick
        return vwx, vwy

    # ── command interface ───────────────────────────────────────────
    def command_joint_velocity(self, cmd) -> None:
        """Set the normalised ``[boom, stick, bucket]`` velocity command (velocity mode)."""
        self.cmd_omega = [_clamp(float(c), -1.0, 1.0) for c in cmd]

    def set_cartesian_target(self, x, y) -> None:
        """Set the wrist position target; IK maps it to joint targets (position mode)."""
        tx = _clamp(x, 0.2, 12.6)
        ty = _clamp(y, 0.1, 7.0)
        shoulder, elbow, cx, cy = geometry.solve_ik(
            tx, ty, self._pivot_x, self._pivot_y, self._boom_l, self._stick_l)
        self.target_wrist_x = cx
        self.target_wrist_y = cy
        self.target_shoulder = _clamp(shoulder, *self._limits[0])
        self.target_elbow = _clamp(elbow, *self._limits[1])

    def nudge_wrist_target(self, dx, dy) -> None:
        self.set_cartesian_target(self.target_wrist_x + dx, self.target_wrist_y + dy)

    def nudge_wrist_rotation(self, dtheta) -> None:
        self.target_wrist = _clamp(self.target_wrist + dtheta, *self._limits[2])

    def set_payload(self, mass) -> None:
        self._payload_mass = mass

    # ── integration ─────────────────────────────────────────────────
    def integrate(self, dt, contact_torques) -> None:
        """Advance joint angles one frame under the hydraulic velocity model.

        ``contact_torques`` is a dict with ``shoulder``/``elbow``/``wrist``
        particle-contact torques (gravity excluded). They drive the load filter
        that throttles joint speed.
        """
        speed = self._load_factors(contact_torques)
        droop = self._payload_droop_factor()
        desired = self._desired_omega(speed, droop)

        for j in range(3):
            lim = self._accel[j] * dt
            diff = desired[j] - self.omega[j]
            self.omega[j] += _clamp(diff, -lim, lim)

        prev = (self.shoulder_angle, self.elbow_angle, self.wrist_angle)
        self.shoulder_angle = _clamp(self.shoulder_angle + self.omega[0] * dt, *self._limits[0])
        self.elbow_angle = _clamp(self.elbow_angle + self.omega[1] * dt, *self._limits[1])
        self.wrist_angle = _clamp(self.wrist_angle + self.omega[2] * dt, *self._limits[2])
        self._enforce_floor(prev)

    # -- Floor (solid bottom wall) constraint -------------------------
    def _lowest_arm_y(self, shoulder, elbow, wrist) -> float:
        """Lowest world-y of the bucket collider (its vertices) + elbow."""
        wx, wy, ex, ey, sax, say, mx, my = geometry.forward_kinematics(
            shoulder, elbow, wrist, self._pivot_x, self._pivot_y,
            self._boom_l, self._stick_l)
        pts, k = geometry.bucket_points(wx, wy, mx, my, sax, say, self._bucket_r)
        low = min(ey, k[1])
        for _, py in pts:
            if py < low:
                low = py
        return low

    def _enforce_floor(self, prev) -> None:
        """Prevent the bucket from passing below the floor.

        If it would penetrate, we try to lift the boom (shoulder_angle) to resolve it,
        which allows the bucket to slide along the floor horizontally when dragging.
        If lifting the boom is insufficient, we fall back to scaling the step.
        """
        new = (self.shoulder_angle, self.elbow_angle, self.wrist_angle)
        if self._lowest_arm_y(*new) >= self._floor_y:
            return

        lo = new[0]
        hi = self._limits[0][1]

        # If even fully raised the arm penetrates, fallback to scaling the whole step.
        if self._lowest_arm_y(hi, new[1], new[2]) < self._floor_y:
            lo_f, hi_f = 0.0, 1.0
            for _ in range(16):
                mid = 0.5 * (lo_f + hi_f)
                a = tuple(prev[j] + (new[j] - prev[j]) * mid for j in range(3))
                if self._lowest_arm_y(*a) >= self._floor_y:
                    lo_f = mid
                else:
                    hi_f = mid
            self.shoulder_angle = prev[0] + (new[0] - prev[0]) * lo_f
            self.elbow_angle = prev[1] + (new[1] - prev[1]) * lo_f
            self.wrist_angle = prev[2] + (new[2] - prev[2]) * lo_f
            return

        # Binary search for the boom angle that puts the lowest point exactly on the floor.
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if self._lowest_arm_y(mid, new[1], new[2]) >= self._floor_y:
                hi = mid
            else:
                lo = mid

        self.shoulder_angle = hi

    def _load_factors(self, contact_torques) -> List[float]:
        """Update the filtered contact load and return per-joint speed factors."""
        keys = ("shoulder", "elbow", "wrist")
        for j, key in enumerate(keys):
            raw = min(1.0, abs(contact_torques.get(key, 0.0)) / max(self.stall_torque[j], 1.0))
            alpha = _ALPHA_RISE if raw > self._filtered_load[j] else _ALPHA_FALL
            self._filtered_load[j] += alpha * (raw - self._filtered_load[j])
        return [max(0.0, 1.0 - load) for load in self._filtered_load]

    def _payload_droop_factor(self) -> float:
        if self._payload_droop <= 0.0:
            return 1.0
        return max(0.3, 1.0 - self._payload_droop * self._payload_mass / 1000.0)

    def _desired_omega(self, speed, droop) -> List[float]:
        max_eff = [self._max_omega[j] * speed[j] * droop for j in range(3)]
        if self.control_mode == "velocity":
            return [self.cmd_omega[j] * max_eff[j] for j in range(3)]
        # position mode: proportional to angle error, capped at the effective max
        errors = (
            self.target_shoulder - self.shoulder_angle,
            self.target_elbow - self.elbow_angle,
            self.target_wrist - self.wrist_angle,
        )
        return [_clamp(self._gain * errors[j], -max_eff[j], max_eff[j]) for j in range(3)]
