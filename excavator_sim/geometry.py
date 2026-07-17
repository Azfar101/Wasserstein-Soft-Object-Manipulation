"""
Pure kinematic geometry: arm forward/inverse kinematics and the bucket profile.

These are stateless functions of scalars only, so both the stateful
:class:`~excavator_sim.arm.ArmDynamics` and the renderer can share them without
a circular import, and the physics solver and renderer derive the bucket shape
from one definition (:data:`excavator_sim.constants.BUCKET_PROFILE`).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .constants import BUCKET_PROFILE, BUCKET_LINK_FRAC


# ── Two-link arm kinematics ─────────────────────────────────────────
def forward_kinematics(shoulder, elbow, wrist, pivot_x, pivot_y, boom_l, stick_l):
    """Full forward kinematics for the boom/stick/bucket chain.

    Returns ``(wrist_x, wrist_y, elbow_x, elbow_y,
    stick_ax, stick_ay, mouth_ax, mouth_ay)`` where ``*_a*`` are unit direction
    vectors of the stick and of the bucket mouth (stick rotated by the wrist
    angle).
    """
    elbow_x = pivot_x + boom_l * math.cos(shoulder)
    elbow_y = pivot_y + boom_l * math.sin(shoulder)

    stick_abs = shoulder - (math.pi - elbow)
    wrist_x = elbow_x + stick_l * math.cos(stick_abs)
    wrist_y = elbow_y + stick_l * math.sin(stick_abs)

    stick_ax = math.cos(stick_abs)
    stick_ay = math.sin(stick_abs)

    cos_w = math.cos(wrist)
    sin_w = math.sin(wrist)
    mouth_ax = stick_ax * cos_w - stick_ay * sin_w
    mouth_ay = stick_ax * sin_w + stick_ay * cos_w

    return (wrist_x, wrist_y, elbow_x, elbow_y,
            stick_ax, stick_ay, mouth_ax, mouth_ay)


def solve_ik(target_x, target_y, pivot_x, pivot_y, boom_l, stick_l):
    """Elbow-up 2-link inverse kinematics.

    Clamps the target into the reachable annulus and returns
    ``(shoulder_angle, elbow_angle, clamped_x, clamped_y)``.
    """
    dx = target_x - pivot_x
    dy = target_y - pivot_y
    raw_d = math.sqrt(dx * dx + dy * dy) + 1e-9

    d_min = abs(boom_l - stick_l) + 0.05
    d_max = boom_l + stick_l - 0.05
    d = max(d_min, min(d_max, raw_d))
    scale = d / raw_d
    cx = pivot_x + dx * scale
    cy = pivot_y + dy * scale

    cos_a = (d * d + boom_l * boom_l - stick_l * stick_l) / (2.0 * d * boom_l)
    cos_a = max(-1.0, min(1.0, cos_a))
    theta = math.atan2(cy - pivot_y, cx - pivot_x)
    shoulder = theta + math.acos(cos_a)

    cos_b = (boom_l * boom_l + stick_l * stick_l - d * d) / (2.0 * boom_l * stick_l)
    cos_b = max(-1.0, min(1.0, cos_b))
    elbow = math.acos(cos_b)

    return shoulder, elbow, cx, cy


# ── Bucket profile ──────────────────────────────────────────────────
def bucket_points(wrist_x, wrist_y, mouth_ax, mouth_ay,
                  link_ax, link_ay, bucket_r):
    """Bucket bowl vertices and the stick-tip connector point.

    Returns ``(pts, k)`` where ``pts = [P0, P1, P2, P3, P4]`` (P0 is the wrist
    pin) and ``k`` is the short connector behind the wrist. Each vertex is
    ``wrist + r * (along * mouth_axis + depth * cavity_axis)`` per
    :data:`BUCKET_PROFILE`.
    """
    cvx, cvy = -mouth_ay, mouth_ax          # cavity axis (into the bowl)
    pts: List[Tuple[float, float]] = [(wrist_x, wrist_y)]
    for along, depth in BUCKET_PROFILE:
        pts.append((
            wrist_x + bucket_r * (along * mouth_ax + depth * cvx),
            wrist_y + bucket_r * (along * mouth_ay + depth * cvy),
        ))
    link_len = bucket_r * BUCKET_LINK_FRAC
    k = (wrist_x - link_len * link_ax, wrist_y - link_len * link_ay)
    return pts, k


def bucket_segments(wrist_x, wrist_y, mouth_ax, mouth_ay,
                    link_ax, link_ay, bucket_r):
    """The 5 collider wall segments of the bucket, as float64 endpoint arrays.

    Returns ``(seg_x0, seg_y0, seg_x1, seg_y1)``, each length 5, for the walls
    P0→P1, P1→P2, P2→P3, P3→P4, K→P0.
    """
    pts, k = bucket_points(wrist_x, wrist_y, mouth_ax, mouth_ay,
                           link_ax, link_ay, bucket_r)
    p0, p1, p2, p3, p4 = pts
    starts = (p0, p1, p2, p3, k)
    ends = (p1, p2, p3, p4, p0)
    seg_x0 = np.array([s[0] for s in starts], dtype=np.float64)
    seg_y0 = np.array([s[1] for s in starts], dtype=np.float64)
    seg_x1 = np.array([e[0] for e in ends], dtype=np.float64)
    seg_y1 = np.array([e[1] for e in ends], dtype=np.float64)
    return seg_x0, seg_y0, seg_x1, seg_y1
