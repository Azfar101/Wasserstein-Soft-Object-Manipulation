"""
Physics-feedback helpers: read the particle field + solver outputs back into
quantities the arm model, observations, and reward need.

* :func:`detect_payload` — soil currently captured inside the bucket bowl.
* :func:`contact_torques` — joint torques from the bucket's reaction force,
  which feed the arm's load-sensing (these are the resistances the soil exerts).
* :func:`boulder_info` — boulder proximity / exposure / burial flags.

These were previously inlined in ``main.py``; centralising them keeps the env
step readable and lets the renderer/HUD reuse the same numbers.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def detect_payload(ps, arm_state: Dict[str, float], bucket_r: float):
    """Mass and centre-of-mass of soil resting inside the bucket bowl.

    Returns ``(total_mass, cm_x, cm_y)``; mass is 0 when the bucket is empty.
    A grain counts as captured when it lies within the wedge swept from the
    wrist along the mouth/cavity axes and inside a bowl-centred radius.
    """
    n = ps.count
    if n == 0:
        return 0.0, 0.0, 0.0

    wx, wy = arm_state["wrist_x"], arm_state["wrist_y"]
    mx, my = arm_state["mouth_ax"], arm_state["mouth_ay"]
    cvx, cvy = -my, mx                       # cavity axis (into the bowl)

    px = ps.px[:n]
    py = ps.py[:n]
    dpx = px - wx
    dpy = py - wy
    along_cavity = dpx * cvx + dpy * cvy
    along_mouth = dpx * mx + dpy * my

    cav_cx = wx + bucket_r * (0.40 * mx + 0.50 * cvx)
    cav_cy = wy + bucket_r * (0.40 * my + 0.50 * cvy)
    dist_cav = np.hypot(px - cav_cx, py - cav_cy)

    inside = ((along_cavity > 0.0) & (along_cavity < bucket_r) &
              (np.abs(along_mouth) < bucket_r * 0.9) &
              (dist_cav < bucket_r))
    if not inside.any():
        return 0.0, 0.0, 0.0

    m = ps.mass[:n][inside]
    total = float(m.sum())
    cx = float((m * px[inside]).sum()) / total
    cy = float((m * py[inside]).sum()) / total
    return total, cx, cy


def contact_torques(arm_state: Dict[str, float], reaction_fx: float,
                    reaction_fy: float, pivot_x: float, pivot_y: float) -> Dict[str, float]:
    """Joint torques from the bucket reaction force (2D cross product r × F).

    The reaction acts at the wrist, so it produces no moment about the wrist
    itself. These are the soil-resistance torques that throttle the arm.
    """
    wx, wy = arm_state["wrist_x"], arm_state["wrist_y"]
    ex, ey = arm_state["elbow_x"], arm_state["elbow_y"]
    return {
        "shoulder": (wx - pivot_x) * reaction_fy - (wy - pivot_y) * reaction_fx,
        "elbow": (wx - ex) * reaction_fy - (wy - ey) * reaction_fx,
        "wrist": 0.0,
    }


def boulder_info(ps, wrist_x: float, wrist_y: float, bucket_r: float,
                 contact_count: int, bury_threshold: int) -> Dict[str, object]:
    """Proximity / exposure / burial summary for the first boulder, if any."""
    n = ps.count
    idxs = np.where(ps.is_boulder[:n])[0]
    if len(idxs) == 0:
        return {"present": False, "rel_x": 0.0, "rel_y": 0.0,
                "exposed": False, "hit": False, "buried": False}

    bi = int(idxs[0])
    bx, by, br = ps.px[bi], ps.py[bi], ps.radius[bi]
    dx = ps.px[:n] - bx
    dy = ps.py[:n] - by
    nearby = int(np.sum(dx * dx + dy * dy < (br * 2.5) ** 2)) - 1  # minus self
    dist = math.hypot(bx - wrist_x, by - wrist_y)
    return {
        "present": True,
        "rel_x": float(bx - wrist_x),
        "rel_y": float(by - wrist_y),
        "exposed": nearby < 4,
        "hit": contact_count > 0 and dist < (bucket_r + br + 0.1),
        "buried": nearby > bury_threshold,
    }
