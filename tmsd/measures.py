"""
External material state as a measure.

The soil bed is summarised as a mass-weighted histogram over the
horizontal axis: μ[i] = (soil mass in column i) / (total soil mass).
Because particles are never created or destroyed inside an episode the
total mass is conserved and the normalized histogram is a probability
distribution on a fixed uniform grid — balanced optimal transport
between two such histograms is well-posed.

The histogram deliberately excludes everything about the robot: no
proprioception, no bucket state. It is the *only* input the TMSD
representation φ ever sees.
"""

from __future__ import annotations

import numpy as np


def soil_mass_histogram(ps, cfg, bins: int = 64) -> np.ndarray:
    """Mass-weighted histogram of soil particles over x, normalized to sum 1.

    Parameters
    ----------
    ps : ParticleSystem
        Live particle state (boulders are excluded via ``soil_mask``).
    cfg : SimConfig
        Provides ``domain_width`` for the fixed binning grid.
    bins : int
        Number of uniform columns across the domain width.

    Returns
    -------
    np.ndarray, shape (bins,), float32, sums to 1 (or zeros if no soil).
    """
    n = ps.count
    h = np.zeros(bins, dtype=np.float64)
    soil = ps.soil_mask
    if n == 0 or not soil.any():
        return h.astype(np.float32)
    px = ps.px[:n][soil]
    mass = ps.mass[:n][soil]
    idx = np.clip((px / cfg.domain_width * bins).astype(np.int64), 0, bins - 1)
    np.add.at(h, idx, mass)
    total = h.sum()
    if total > 0.0:
        h /= total
    return h.astype(np.float32)


def bin_centers(cfg, bins: int = 64) -> np.ndarray:
    """World-x coordinates (meters) of the histogram bin centers."""
    edges = np.linspace(0.0, cfg.domain_width, bins + 1)
    return ((edges[:-1] + edges[1:]) * 0.5).astype(np.float64)


def soil_mass_histogram_2d(ps, cfg, bins_x: int = 48, bins_y: int = 24) -> np.ndarray:
    """2D mass histogram over (x, y), flattened row-major, normalized to 1.

    Unlike the 1D x-marginal, this sees vertical structure: a deep
    narrow hole and a wide shallow one are different states here.
    """
    k = bins_x * bins_y
    h = np.zeros(k, dtype=np.float64)
    n = ps.count
    soil = ps.soil_mask
    if n == 0 or not soil.any():
        return h.astype(np.float32)
    px = ps.px[:n][soil]
    py = ps.py[:n][soil]
    mass = ps.mass[:n][soil]
    ix = np.clip((px / cfg.domain_width * bins_x).astype(np.int64), 0, bins_x - 1)
    iy = np.clip((py / cfg.domain_height * bins_y).astype(np.int64), 0, bins_y - 1)
    np.add.at(h, ix * bins_y + iy, mass)
    total = h.sum()
    if total > 0.0:
        h /= total
    return h.astype(np.float32)


def bucket_payload_mask(ps, arm_state, bucket_r: float) -> np.ndarray:
    """Boolean mask (length ``count``) of soil grains inside the bucket bowl.

    Mirrors :func:`excavator_sim.physics_feedback.detect_payload`'s wedge
    test, but returns the mask so measures can partition mass by channel.
    """
    n = ps.count
    if n == 0:
        return np.zeros(0, dtype=bool)
    wx, wy = arm_state["wrist_x"], arm_state["wrist_y"]
    mx, my = arm_state["mouth_ax"], arm_state["mouth_ay"]
    cvx, cvy = -my, mx
    px, py = ps.px[:n], ps.py[:n]
    dpx, dpy = px - wx, py - wy
    along_cavity = dpx * cvx + dpy * cvy
    along_mouth = dpx * mx + dpy * my
    cav_cx = wx + bucket_r * (0.40 * mx + 0.50 * cvx)
    cav_cy = wy + bucket_r * (0.40 * my + 0.50 * cvy)
    dist_cav = np.hypot(px - cav_cx, py - cav_cy)
    return ((along_cavity > 0.0) & (along_cavity < bucket_r) &
            (np.abs(along_mouth) < bucket_r * 0.9) &
            (dist_cav < bucket_r) & ps.soil_mask)


AIRBORNE_SPEED = 1.0   # m/s: faster soil counts as detached / in flight


def soil_state_composite(ps, cfg, arm_state, bucket_r: float,
                         bins: int = 64, air_bins: int = 16) -> np.ndarray:
    """Composite external state: soil mass partitioned into three channels.

      [ ground mass histogram over x   (bins,   mass fraction)
        bucket payload: mass fraction, cm_x/W, cm_y/H          (3,)
        airborne mass histogram over x (air_bins, mass fraction) ]

    Ground + bucket + airborne mass fractions sum to 1: the channels
    partition the same conserved mass, so "soil in the bucket" and
    "soil in flight" are first-class, φ-visible coordinates — without
    them, carrying is indistinguishable from ground transport and
    carry/dump skills cannot form.
    """
    n = ps.count
    out = np.zeros(bins + 3 + air_bins, dtype=np.float64)
    soil = ps.soil_mask
    if n == 0 or not soil.any():
        return out.astype(np.float32)
    px, py = ps.px[:n], ps.py[:n]
    vx, vy = ps.vx[:n], ps.vy[:n]
    mass = ps.mass[:n]
    total = mass[soil].sum()

    in_bucket = bucket_payload_mask(ps, arm_state, bucket_r)
    speed = np.hypot(vx, vy)
    airborne = soil & ~in_bucket & (speed > AIRBORNE_SPEED)
    ground = soil & ~in_bucket & ~airborne

    if ground.any():
        idx = np.clip((px[ground] / cfg.domain_width * bins).astype(np.int64),
                      0, bins - 1)
        np.add.at(out[:bins], idx, mass[ground])
    if in_bucket.any():
        bm = mass[in_bucket]
        out[bins] = bm.sum()
        out[bins + 1] = float((bm * px[in_bucket]).sum()) / bm.sum() / cfg.domain_width
        out[bins + 2] = float((bm * py[in_bucket]).sum()) / bm.sum() / cfg.domain_height
    if airborne.any():
        idx = np.clip((px[airborne] / cfg.domain_width * air_bins).astype(np.int64),
                      0, air_bins - 1)
        np.add.at(out[bins + 3:], idx, mass[airborne])

    out[:bins] /= total
    out[bins] /= total
    out[bins + 3:] /= total
    return out.astype(np.float32)


def bin_centers_2d(cfg, bins_x: int = 48, bins_y: int = 24) -> np.ndarray:
    """(bins_x*bins_y, 2) world coordinates of 2D bin centers (row-major,
    matching :func:`soil_mass_histogram_2d`)."""
    ex = np.linspace(0.0, cfg.domain_width, bins_x + 1)
    ey = np.linspace(0.0, cfg.domain_height, bins_y + 1)
    cx = (ex[:-1] + ex[1:]) * 0.5
    cy = (ey[:-1] + ey[1:]) * 0.5
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float64)
