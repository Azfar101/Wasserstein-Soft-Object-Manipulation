"""
CPU physics kernels (Numba ``@njit``, parallelised with ``prange``).

Implements the 2D Hertz-Mindlin DEM contact model, boundary walls, semi-implicit
Euler integration, and the segment-wall bucket collider. The CUDA twins in
:mod:`excavator_sim.cuda_backend` mirror these exactly; keep the two numerically
in lock-step when editing.

Physics
-------
* Normal force (Hertz): ``F_n = (4/3) E* sqrt(R*) δ^{3/2}``.
* Normal damping: ``F_nd = 2 β sqrt(5/6 · S_n · m*) · v_n`` with
  ``S_n = 2 E* sqrt(R* δ)`` and β from the restitution.
* Tangential force: velocity-based, Coulomb-limited ``|F_t| ≤ μ |F_n|``.
* Rolling resistance: applied only when supported from below.
* Cohesion (optional): constant attraction within a small cutoff.
"""

import math
import numpy as np
from numba import njit, prange

from .constants import (
    POISSON_RATIO, MAX_OVERLAP_RATIO, GLOBAL_DAMPING, MAX_VELOCITY,
    CELL_SIZE, BUCKET_WALL_T, BUCKET_RESTITUTION, BUCKET_BAUMGARTE,
)
from .spatial_hash import GRID_COLS, GRID_ROWS


# ── Pairwise contact ────────────────────────────────────────────────
@njit(cache=True)
def hertz_pair(xi, yi, vxi, vyi, ri, mi,
               xj, yj, vxj, vyj, rj, mj,
               e_star, mu_i, mu_j, beta, cohesion_on, cohesion_str):
    """Contact force on particle *i* from particle *j*. Returns ``(fx, fy)``.

    Effective friction is the mean of the two coefficients.
    """
    mu = (mu_i + mu_j) * 0.5
    dx = xj - xi
    dy = yj - yi
    dist = math.sqrt(dx * dx + dy * dy)
    overlap = ri + rj - dist

    if overlap <= 0.0:
        # Optional cohesion: weak attraction just outside contact.
        if cohesion_on and dist < (ri + rj) * 1.5:
            mag = cohesion_str * overlap          # overlap < 0 → attractive
            inv = 1.0 / max(dist, 1e-12)
            return mag * dx * inv, mag * dy * inv
        return 0.0, 0.0

    inv_dist = 1.0 / max(dist, 1e-12)
    nx = dx * inv_dist
    ny = dy * inv_dist

    r_star = (ri * rj) / (ri + rj)
    m_star = (mi * mj) / (mi + mj)

    max_overlap = MAX_OVERLAP_RATIO * r_star
    if overlap > max_overlap:
        overlap = max_overlap

    sqrt_r_star = math.sqrt(r_star)
    sqrt_delta = math.sqrt(overlap)

    # Normal force (Hertz) + velocity damping.
    kn = (4.0 / 3.0) * e_star * sqrt_r_star
    fn_elastic = kn * overlap * sqrt_delta
    sn = 2.0 * e_star * math.sqrt(r_star * overlap)
    dn = 2.0 * beta * math.sqrt((5.0 / 6.0) * sn * m_star)

    dvx = vxi - vxj
    dvy = vyi - vyj
    vn = dvx * nx + dvy * ny
    fn_total = fn_elastic + dn * vn
    if fn_total < 0.0:
        fn_total = 0.0

    # Tangential (Coulomb-limited) friction.
    tx = -ny
    ty = nx
    vt = dvx * tx + dvy * ty
    st = 2.0 * (1.0 - POISSON_RATIO) / (2.0 - POISSON_RATIO) * sn
    dt = 2.0 * beta * math.sqrt(max((5.0 / 6.0) * st * m_star, 0.0))
    ft_limit = mu * fn_total
    ft = min(abs(vt) * dt, ft_limit)
    if vt > 0.0:
        ft = -ft

    return -fn_total * nx + ft * tx, -fn_total * ny + ft * ty


@njit(cache=True)
def boundary_force(xi, yi, vxi, vyi, ri, mi, e_star, beta, mu, domain_w, domain_h):
    """Contact with the four static walls (floor, ceiling, left, right)."""
    fx = 0.0
    fy = 0.0
    for w in range(4):
        if w == 0:
            wnx, wny, d = 0.0, 1.0, yi - ri               # floor
        elif w == 1:
            wnx, wny, d = 0.0, -1.0, domain_h - yi - ri    # ceiling
        elif w == 2:
            wnx, wny, d = 1.0, 0.0, xi - ri               # left
        else:
            wnx, wny, d = -1.0, 0.0, domain_w - xi - ri    # right

        if d >= 0.0:
            continue
        overlap = -d
        max_overlap = MAX_OVERLAP_RATIO * ri
        if overlap > max_overlap:
            overlap = max_overlap
        sqrt_delta = math.sqrt(overlap)
        sqrt_r_star = math.sqrt(ri)              # wall: R*=ri, m*=mi

        kn = (4.0 / 3.0) * e_star * sqrt_r_star
        fn_elastic = kn * overlap * sqrt_delta
        sn = 2.0 * e_star * math.sqrt(ri * overlap)
        dn = 2.0 * beta * math.sqrt((5.0 / 6.0) * sn * mi)
        vn = vxi * wnx + vyi * wny               # >0 separating, <0 approaching
        fn_total = fn_elastic - dn * vn
        if fn_total < 0.0:
            fn_total = 0.0

        tx = -wny
        ty = wnx
        vt = vxi * tx + vyi * ty
        st = 2.0 * (1.0 - POISSON_RATIO) / (2.0 - POISSON_RATIO) * sn
        dt = 2.0 * beta * math.sqrt(max((5.0 / 6.0) * st * mi, 0.0))
        ft_limit = mu * fn_total
        ft = min(abs(vt) * dt, ft_limit)
        if vt > 0.0:
            ft = -ft

        fx += fn_total * wnx + ft * tx
        fy += fn_total * wny + ft * ty
    return fx, fy


@njit(cache=True, parallel=True)
def solve_forces(px, py, vx, vy, ax, ay, radius, mass, inv_mass, n,
                 cell_counts, cell_particles,
                 e_star, mu, beta, gravity, domain_w, domain_h,
                 cohesion_on, cohesion_str, rolling_res,
                 soil_r_max, is_boulder, boul_indices, n_boulders, friction):
    """Accumulate per-particle accelerations using the spatial-hash grid.

    Boulders are excluded from the grid and handled by an explicit all-boulder
    check, so the per-particle grid search stays at the soil radius even when a
    large boulder is present.
    """
    for i in prange(n):
        fxi = 0.0
        fyi = -gravity * mass[i]
        is_supported = False

        # Particle–particle contacts via grid neighbours.
        search_r = int((radius[i] + soil_r_max) / CELL_SIZE) + 1
        ci_x = int(px[i] / CELL_SIZE)
        ci_y = int(py[i] / CELL_SIZE)
        for dy in range(-search_r, search_r + 1):
            ny_ = ci_y + dy
            if ny_ < 0 or ny_ >= GRID_ROWS:
                continue
            for dx in range(-search_r, search_r + 1):
                nx_ = ci_x + dx
                if nx_ < 0 or nx_ >= GRID_COLS:
                    continue
                ci = ny_ * GRID_COLS + nx_
                for k in range(cell_counts[ci]):
                    j = cell_particles[ci, k]
                    if j == i or is_boulder[j]:
                        continue
                    if py[j] < py[i] and (px[j] - px[i]) ** 2 + (py[j] - py[i]) ** 2 \
                            < (radius[i] + radius[j]) ** 2:
                        is_supported = True
                    cfx, cfy = hertz_pair(
                        px[i], py[i], vx[i], vy[i], radius[i], mass[i],
                        px[j], py[j], vx[j], vy[j], radius[j], mass[j],
                        e_star, friction[i], friction[j], beta,
                        cohesion_on, cohesion_str)
                    fxi += cfx
                    fyi += cfy

        # Explicit boulder contacts.
        for bi in range(n_boulders):
            j = boul_indices[bi]
            if j == i:
                continue
            if py[j] < py[i] and (px[j] - px[i]) ** 2 + (py[j] - py[i]) ** 2 \
                    < (radius[i] + radius[j]) ** 2:
                is_supported = True
            cfx, cfy = hertz_pair(
                px[i], py[i], vx[i], vy[i], radius[i], mass[i],
                px[j], py[j], vx[j], vy[j], radius[j], mass[j],
                e_star, friction[i], friction[j], beta,
                cohesion_on, cohesion_str)
            fxi += cfx
            fyi += cfy

        # Boundary walls.
        bfx, bfy = boundary_force(px[i], py[i], vx[i], vy[i], radius[i], mass[i],
                                  e_star, beta, mu, domain_w, domain_h)
        fxi += bfx
        fyi += bfy
        if bfy > 0.0:
            is_supported = True

        # Rolling resistance — only when supported from below.
        if is_supported:
            speed = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
            if speed > 1e-8:
                rr_mag = rolling_res * mass[i] * gravity
                fxi -= rr_mag * vx[i] / speed
                fyi -= rr_mag * vy[i] / speed

        ax[i] += fxi * inv_mass[i]
        ay[i] += fyi * inv_mass[i]


@njit(cache=True, parallel=True)
def integrate(px, py, vx, vy, ax, ay, n, dt, domain_w, domain_h, radius):
    """Semi-implicit Euler with mild damping, a speed cap, and domain clamping."""
    for i in prange(n):
        vx[i] = (vx[i] + ax[i] * dt) * GLOBAL_DAMPING
        vy[i] = (vy[i] + ay[i] * dt) * GLOBAL_DAMPING

        speed = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
        if speed > MAX_VELOCITY:
            scale = MAX_VELOCITY / speed
            vx[i] *= scale
            vy[i] *= scale

        px[i] += vx[i] * dt
        py[i] += vy[i] * dt

        r = radius[i]
        if px[i] < r:
            px[i] = r
            vx[i] = 0.0
        elif px[i] > domain_w - r:
            px[i] = domain_w - r
            vx[i] = 0.0
        if py[i] < r:
            py[i] = r
            vy[i] = 0.0
        elif py[i] > domain_h - r:
            py[i] = domain_h - r
            vy[i] = 0.0


@njit(cache=True)
def count_boulder_neighbors(px, py, radius, is_boulder, n, boul_idx, margin):
    """Count soil particles overlapping or nearly touching a boulder."""
    if boul_idx < 0:
        return 0
    bx = px[boul_idx]
    by = py[boul_idx]
    br = radius[boul_idx]
    count = 0
    for i in range(n):
        if i == boul_idx or is_boulder[i]:
            continue
        dx = px[i] - bx
        dy = py[i] - by
        touch_r = br + radius[i] + margin
        if dx * dx + dy * dy < touch_r * touch_r:
            count += 1
    return count


# ── Excavator bucket ────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def apply_bucket_scoop(px, py, vx, vy, radius, mass, inv_mass, n,
                       seg_x0, seg_y0, seg_x1, seg_y1,
                       wrist_vx, wrist_vy, e_star, mu, dt_sub, out_jx, out_jy):
    """Segment-wall bucket collider with Coulomb tangential friction.

    The 5 bucket walls are passed in as precomputed segment endpoints (see
    :func:`excavator_sim.geometry.bucket_segments`). Each particle is resolved
    against every wall via Baumgarte position correction, velocity reflection,
    and a Hertz-limited tangential impulse. ``out_jx/out_jy`` receive the impulse
    the bucket applied to each particle this substep (the caller sums them and
    negates for the bucket reaction).
    """
    for i in prange(n):
        ri = radius[i]
        pi_x = px[i]
        pi_y = py[i]
        vi_x = vx[i]
        vi_y = vy[i]
        vi_x0 = vi_x
        vi_y0 = vi_y
        inv_mi = inv_mass[i]
        sqrt_ri = math.sqrt(ri)
        thr = BUCKET_WALL_T + ri
        hit = False

        for s in range(5):
            s0x = seg_x0[s]
            s0y = seg_y0[s]
            edx = seg_x1[s] - s0x
            edy = seg_y1[s] - s0y
            seg_sq = edx * edx + edy * edy + 1e-12
            t = ((pi_x - s0x) * edx + (pi_y - s0y) * edy) / seg_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            dx_c = pi_x - (s0x + t * edx)
            dy_c = pi_y - (s0y + t * edy)
            dist = math.sqrt(dx_c * dx_c + dy_c * dy_c)
            if dist < thr and dist > 1e-9:
                nx = dx_c / dist
                ny = dy_c / dist
                ov = thr - dist
                pi_x += nx * ov * BUCKET_BAUMGARTE
                pi_y += ny * ov * BUCKET_BAUMGARTE
                vn_rel = (vi_x - wrist_vx) * nx + (vi_y - wrist_vy) * ny
                if vn_rel < 0.0:
                    imp = -(1.0 + BUCKET_RESTITUTION) * vn_rel
                    vi_x += imp * nx
                    vi_y += imp * ny
                tx = -ny
                ty = nx
                vt_rel = (vi_x - wrist_vx) * tx + (vi_y - wrist_vy) * ty
                fn_hertz = (4.0 / 3.0) * e_star * sqrt_ri * ov * math.sqrt(ov)
                ft_max = mu * fn_hertz * dt_sub * inv_mi
                ft_imp = min(abs(vt_rel), ft_max)
                if vt_rel > 0.0:
                    ft_imp = -ft_imp
                vi_x += ft_imp * tx
                vi_y += ft_imp * ty
                hit = True

        if hit:
            px[i] = pi_x
            py[i] = pi_y
            vx[i] = vi_x
            vy[i] = vi_y
            out_jx[i] += mass[i] * (vi_x - vi_x0)
            out_jy[i] += mass[i] * (vi_y - vi_y0)
