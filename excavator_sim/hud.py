"""
Text builders for the on-screen overlay and for structured state logging.

Kept separate from the renderer so the same numbers can be printed to a terminal
or dumped to a training log without a display.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def compact_overlay(info: Dict, fps: float = 0.0) -> List[str]:
    """A short status overlay for the env's own ``render()`` path."""
    wx, wy = info.get("wrist_xy", (0.0, 0.0))
    ct = info.get("contact_torques", {})
    lines = [
        f"FPS {fps:4.0f}   backend {info.get('backend', '?').upper()}   "
        f"contacts {info.get('contact_count', 0)}",
        f"wrist ({wx:4.1f}, {wy:4.1f})   payload {info.get('payload_mass', 0.0):6.1f} kg"
        f"   lifted {info.get('lifted_kg', 0.0):6.1f} kg",
        f"torque  S {ct.get('shoulder', 0.0):+8.0f}  E {ct.get('elbow', 0.0):+8.0f}  "
        f"N·m",
    ]
    if info.get("is_success"):
        lines.insert(0, "*** SUCCESS ***")
    b = info.get("boulder", {})
    if b.get("present"):
        if b.get("exposed"):
            lines.insert(0, "*** BOULDER EXPOSED ***")
        elif b.get("hit"):
            lines.insert(0, "!! BOULDER CONTACT !!")
    return lines


def state_dict(env, fps: float = 0.0) -> Dict[str, object]:
    """Flat, JSON-friendly snapshot of the whole sim — handy for logging / VLA."""
    ps = env.ps
    n = ps.count
    soil = ps.soil_mask
    st = env.arm_state
    cfg = env.cfg
    speeds = np.hypot(ps.vx[:n], ps.vy[:n]) if n else np.zeros(1)

    return {
        "fps": round(fps, 1),
        "backend": "cuda" if env.solver.use_cuda else "cpu",
        "steps": env.steps,
        "particle_count": n,
        "soil_count": int(soil.sum()),
        "boulder_count": int((~soil).sum()),
        # soil parameters
        "gravity": round(cfg.soil.gravity, 3),
        "friction": round(cfg.soil.friction, 3),
        "youngs_modulus": cfg.soil.youngs_modulus,
        "cohesion": cfg.soil.cohesion,
        "cohesion_strength": round(cfg.soil.cohesion_strength, 1),
        "density": round(cfg.particle.density, 1),
        # kinematics
        "mean_speed": round(float(speeds.mean()), 4) if n else 0.0,
        "max_speed": round(float(speeds.max()), 4) if n else 0.0,
        # arm
        "shoulder_angle": round(st["shoulder_angle"], 4),
        "elbow_angle": round(st["elbow_angle"], 4),
        "wrist_angle": round(st["wrist_angle"], 4),
        "omega": [round(st["omega_shoulder"], 4), round(st["omega_elbow"], 4),
                  round(st["omega_wrist"], 4)],
        "wrist_xy": [round(st["wrist_x"], 4), round(st["wrist_y"], 4)],
        "load": [round(st["load_shoulder"], 3), round(st["load_elbow"], 3),
                 round(st["load_wrist"], 3)],
        "payload_kg": round(env.payload_mass, 2),
        "contact_torques": {k: round(v, 1) for k, v in env.contact_torques.items()},
        "contact_count": env.solver.bucket_contact_count,
        "bucket_reaction": [round(env.solver.bucket_reaction_fx, 1),
                            round(env.solver.bucket_reaction_fy, 1)],
        "boulder": env.boulder,
    }
