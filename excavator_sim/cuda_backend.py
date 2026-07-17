"""
CUDA physics kernels — GPU mirror of :mod:`excavator_sim.cpu_backend`.

All device math is float32: consumer GPUs run FP64 at ~1/64 the FP32 rate, so
float32 kernels with named ``np.float32`` constants (to stop Python literals
upcasting expressions to float64) are dramatically faster. Host arrays stay
float64; the solver casts on transfer.

Kernels are defined unconditionally (Numba compiles them lazily on first call,
which needs no GPU at import time). :func:`init_cuda` performs the CUDA 13.x DLL
path fix-up and a probe compile, and reports whether the GPU path is usable.
"""

import os
import glob
import math

import numpy as np
from numba import cuda

from .constants import (
    MAX_OVERLAP_RATIO, GLOBAL_DAMPING, MAX_VELOCITY,
    BUCKET_WALL_T, BUCKET_RESTITUTION, BUCKET_BAUMGARTE,
)
from .spatial_hash import GRID_COLS, GRID_ROWS, GRID_TOTAL, MAX_PER_CELL

# Set True by a successful init_cuda(); the Solver reads it.
CUDA_AVAILABLE = False


# ── float32 constants (keep device expressions in FP32) ─────────────
_F32_ZERO = np.float32(0.0)
_F32_ONE = np.float32(1.0)
_F32_TWO = np.float32(2.0)
_F32_HALF = np.float32(0.5)
_F32_4_3 = np.float32(4.0 / 3.0)
_F32_5_6 = np.float32(5.0 / 6.0)
_F32_1_5 = np.float32(1.5)
_F32_POISSON_F1 = np.float32(1.0 - 0.3)     # 1 − ν
_F32_POISSON_F2 = np.float32(2.0 - 0.3)     # 2 − ν
_F32_MAX_OV = np.float32(MAX_OVERLAP_RATIO)
_F32_EPS = np.float32(1e-9)
_F32_DAMPING = np.float32(GLOBAL_DAMPING)
_F32_MAX_V = np.float32(MAX_VELOCITY)
_F32_WALL = np.float32(BUCKET_WALL_T)
_F32_REST = np.float32(BUCKET_RESTITUTION)
_F32_BAUM = np.float32(BUCKET_BAUMGARTE)


# ── Device contact functions ────────────────────────────────────────
@cuda.jit(device=True)
def _hertz_pair(xi, yi, vxi, vyi, ri, mi,
                xj, yj, vxj, vyj, rj, mj,
                e_star, mu_i, mu_j, beta, cohesion_on, cohesion_str):
    mu = (mu_i + mu_j) * _F32_HALF
    dx = xj - xi
    dy = yj - yi
    dist = math.sqrt(dx * dx + dy * dy)
    overlap = ri + rj - dist
    if overlap <= _F32_ZERO:
        if cohesion_on and dist < (ri + rj) * _F32_1_5:
            safe_d = dist if dist > _F32_EPS else _F32_EPS
            mag = cohesion_str * overlap
            return mag * dx / safe_d, mag * dy / safe_d
        return _F32_ZERO, _F32_ZERO

    safe_d = dist if dist > _F32_EPS else _F32_EPS
    inv_dist = _F32_ONE / safe_d
    nx = dx * inv_dist
    ny = dy * inv_dist
    r_star = (ri * rj) / (ri + rj)
    m_star = (mi * mj) / (mi + mj)
    max_ov = _F32_MAX_OV * r_star
    if overlap > max_ov:
        overlap = max_ov
    sqrt_r_star = math.sqrt(r_star)
    sqrt_delta = math.sqrt(overlap)
    kn = _F32_4_3 * e_star * sqrt_r_star
    fn_elastic = kn * overlap * sqrt_delta
    sn = _F32_TWO * e_star * math.sqrt(r_star * overlap)
    dn = _F32_TWO * beta * math.sqrt(_F32_5_6 * sn * m_star)
    dvx = vxi - vxj
    dvy = vyi - vyj
    vn = dvx * nx + dvy * ny
    fn_total = fn_elastic + dn * vn
    if fn_total < _F32_ZERO:
        fn_total = _F32_ZERO
    tx = -ny
    ty = nx
    vt = dvx * tx + dvy * ty
    st = _F32_TWO * _F32_POISSON_F1 / _F32_POISSON_F2 * sn
    dt = _F32_TWO * beta * math.sqrt(max(_F32_5_6 * st * m_star, _F32_ZERO))
    ft_limit = mu * fn_total
    ft = min(abs(vt) * dt, ft_limit)
    if vt > _F32_ZERO:
        ft = -ft
    return -fn_total * nx + ft * tx, -fn_total * ny + ft * ty


@cuda.jit(device=True)
def _boundary(xi, yi, vxi, vyi, ri, mi, e_star, beta, mu, domain_w, domain_h):
    fx = _F32_ZERO
    fy = _F32_ZERO
    for w in range(4):
        if w == 0:
            wnx = _F32_ZERO; wny = _F32_ONE; d = yi - ri
        elif w == 1:
            wnx = _F32_ZERO; wny = -_F32_ONE; d = domain_h - yi - ri
        elif w == 2:
            wnx = _F32_ONE; wny = _F32_ZERO; d = xi - ri
        else:
            wnx = -_F32_ONE; wny = _F32_ZERO; d = domain_w - xi - ri
        if d >= _F32_ZERO:
            continue
        overlap = -d
        max_ov = _F32_MAX_OV * ri
        if overlap > max_ov:
            overlap = max_ov
        sqrt_delta = math.sqrt(overlap)
        sqrt_r_star = math.sqrt(ri)
        kn = _F32_4_3 * e_star * sqrt_r_star
        fn_elastic = kn * overlap * sqrt_delta
        sn = _F32_TWO * e_star * math.sqrt(ri * overlap)
        dn = _F32_TWO * beta * math.sqrt(_F32_5_6 * sn * mi)
        vn = vxi * wnx + vyi * wny
        fn_total = fn_elastic - dn * vn
        if fn_total < _F32_ZERO:
            fn_total = _F32_ZERO
        tx = -wny
        ty = wnx
        vt = vxi * tx + vyi * ty
        st = _F32_TWO * _F32_POISSON_F1 / _F32_POISSON_F2 * sn
        dt = _F32_TWO * beta * math.sqrt(max(_F32_5_6 * st * mi, _F32_ZERO))
        ft_limit = mu * fn_total
        ft = min(abs(vt) * dt, ft_limit)
        if vt > _F32_ZERO:
            ft = -ft
        fx += fn_total * wnx + ft * tx
        fy += fn_total * wny + ft * ty
    return fx, fy


@cuda.jit(device=True)
def _segment_collide(pi_x, pi_y, vi_x, vi_y, ri, sqrt_ri, inv_mi,
                     s0x, s0y, s1x, s1y, wrist_vx, wrist_vy, e_star, mu, dt_sub):
    """Resolve one particle against one bucket wall segment.

    Returns ``(pi_x, pi_y, vi_x, vi_y, hit)`` — the (possibly corrected) state
    and whether contact occurred. Replaces the five copy-pasted blocks the old
    kernel carried.
    """
    edx = s1x - s0x
    edy = s1y - s0y
    seg_sq = edx * edx + edy * edy + _F32_EPS
    t = ((pi_x - s0x) * edx + (pi_y - s0y) * edy) / seg_sq
    if t < _F32_ZERO:
        t = _F32_ZERO
    elif t > _F32_ONE:
        t = _F32_ONE
    dx_c = pi_x - (s0x + t * edx)
    dy_c = pi_y - (s0y + t * edy)
    dist = math.sqrt(dx_c * dx_c + dy_c * dy_c)
    thr = _F32_WALL + ri
    hit = False
    if dist < thr and dist > _F32_EPS:
        nx = dx_c / dist
        ny = dy_c / dist
        ov = thr - dist
        pi_x += nx * ov * _F32_BAUM
        pi_y += ny * ov * _F32_BAUM
        vn = (vi_x - wrist_vx) * nx + (vi_y - wrist_vy) * ny
        if vn < _F32_ZERO:
            imp = -(_F32_ONE + _F32_REST) * vn
            vi_x += imp * nx
            vi_y += imp * ny
        tx = -ny
        ty = nx
        vt = (vi_x - wrist_vx) * tx + (vi_y - wrist_vy) * ty
        fn_hertz = _F32_4_3 * e_star * sqrt_ri * ov * math.sqrt(ov)
        ft_max = mu * fn_hertz * dt_sub * inv_mi
        ft = min(abs(vt), ft_max)
        if vt > _F32_ZERO:
            ft = -ft
        vi_x += ft * tx
        vi_y += ft * ty
        hit = True
    return pi_x, pi_y, vi_x, vi_y, hit


# ── Kernels ─────────────────────────────────────────────────────────
@cuda.jit
def build_grid_kernel(px, py, n, cell_counts, cell_particles,
                      grid_cols, grid_rows, cell_size, max_per_cell, is_boulder):
    """Bin non-boulder particles into the spatial-hash grid (atomic counts)."""
    i = cuda.grid(1)
    if i >= n or is_boulder[i]:
        return
    col = int(px[i] / cell_size)
    row = int(py[i] / cell_size)
    if col < 0:
        col = 0
    elif col >= grid_cols:
        col = grid_cols - 1
    if row < 0:
        row = 0
    elif row >= grid_rows:
        row = grid_rows - 1
    ci = row * grid_cols + col
    slot = cuda.atomic.add(cell_counts, ci, 1)
    if slot < max_per_cell:
        cell_particles[ci, slot] = i


@cuda.jit
def zero_f32_kernel(arr, n):
    i = cuda.grid(1)
    if i < n:
        arr[i] = _F32_ZERO


@cuda.jit
def zero_i32_kernel(arr, n):
    i = cuda.grid(1)
    if i < n:
        arr[i] = 0


@cuda.jit
def solve_kernel(px, py, vx, vy, ax, ay, radius, mass, inv_mass, n,
                 cell_counts, cell_particles,
                 e_star, mu, beta, gravity, domain_w, domain_h,
                 cohesion_on, cohesion_str, rolling_res,
                 grid_cols, grid_rows, cell_size, max_per_cell,
                 soil_r_max, boul_indices, n_boulders, friction):
    i = cuda.grid(1)
    if i >= n:
        return
    fxi = _F32_ZERO
    fyi = -gravity * mass[i]
    is_supported = False

    search_r = int((radius[i] + soil_r_max) / cell_size) + 1
    ci_x = int(px[i] / cell_size)
    ci_y = int(py[i] / cell_size)
    for dy in range(-search_r, search_r + 1):
        ny_ = ci_y + dy
        if ny_ < 0 or ny_ >= grid_rows:
            continue
        for dx in range(-search_r, search_r + 1):
            nx_ = ci_x + dx
            if nx_ < 0 or nx_ >= grid_cols:
                continue
            ci = ny_ * grid_cols + nx_
            for k in range(cell_counts[ci]):
                j = cell_particles[ci, k]
                if j == i:
                    continue
                if py[j] < py[i] and (px[j] - px[i]) ** 2 + (py[j] - py[i]) ** 2 \
                        < (radius[i] + radius[j]) ** 2:
                    is_supported = True
                cfx, cfy = _hertz_pair(
                    px[i], py[i], vx[i], vy[i], radius[i], mass[i],
                    px[j], py[j], vx[j], vy[j], radius[j], mass[j],
                    e_star, friction[i], friction[j], beta, cohesion_on, cohesion_str)
                fxi += cfx
                fyi += cfy

    for bi in range(n_boulders):
        j = boul_indices[bi]
        if j == i:
            continue
        if py[j] < py[i] and (px[j] - px[i]) ** 2 + (py[j] - py[i]) ** 2 \
                < (radius[i] + radius[j]) ** 2:
            is_supported = True
        cfx, cfy = _hertz_pair(
            px[i], py[i], vx[i], vy[i], radius[i], mass[i],
            px[j], py[j], vx[j], vy[j], radius[j], mass[j],
            e_star, friction[i], friction[j], beta, cohesion_on, cohesion_str)
        fxi += cfx
        fyi += cfy

    bfx, bfy = _boundary(px[i], py[i], vx[i], vy[i], radius[i], mass[i],
                         e_star, beta, mu, domain_w, domain_h)
    fxi += bfx
    fyi += bfy
    if bfy > _F32_ZERO:
        is_supported = True

    if is_supported:
        speed = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
        if speed > _F32_EPS:
            rr_mag = rolling_res * mass[i] * gravity
            fxi -= rr_mag * vx[i] / speed
            fyi -= rr_mag * vy[i] / speed

    ax[i] = fxi * inv_mass[i]
    ay[i] = fyi * inv_mass[i]


@cuda.jit
def integrate_kernel(px, py, vx, vy, ax, ay, n, dt, domain_w, domain_h, radius):
    i = cuda.grid(1)
    if i >= n:
        return
    vx[i] = (vx[i] + ax[i] * dt) * _F32_DAMPING
    vy[i] = (vy[i] + ay[i] * dt) * _F32_DAMPING
    speed = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
    if speed > _F32_MAX_V:
        scale = _F32_MAX_V / speed
        vx[i] *= scale
        vy[i] *= scale
    px[i] += vx[i] * dt
    py[i] += vy[i] * dt
    r = radius[i]
    if px[i] < r:
        px[i] = r
        vx[i] = _F32_ZERO
    elif px[i] > domain_w - r:
        px[i] = domain_w - r
        vx[i] = _F32_ZERO
    if py[i] < r:
        py[i] = r
        vy[i] = _F32_ZERO
    elif py[i] > domain_h - r:
        py[i] = domain_h - r
        vy[i] = _F32_ZERO


@cuda.jit
def bucket_scoop_kernel(px, py, vx, vy, radius, mass, inv_mass, n,
                        seg_x0, seg_y0, seg_x1, seg_y1,
                        wrist_vx, wrist_vy, e_star, mu, dt_sub,
                        accum_jx, accum_jy):
    """Segment-wall bucket collider; reaction impulse accumulated via atomics."""
    i = cuda.grid(1)
    if i >= n:
        return
    ri = radius[i]
    sqrt_ri = math.sqrt(ri)
    inv_mi = inv_mass[i]
    pi_x = px[i]
    pi_y = py[i]
    vi_x = vx[i]
    vi_y = vy[i]
    vi_x0 = vi_x
    vi_y0 = vi_y
    hit_any = False
    for s in range(5):
        pi_x, pi_y, vi_x, vi_y, hit = _segment_collide(
            pi_x, pi_y, vi_x, vi_y, ri, sqrt_ri, inv_mi,
            seg_x0[s], seg_y0[s], seg_x1[s], seg_y1[s],
            wrist_vx, wrist_vy, e_star, mu, dt_sub)
        if hit:
            hit_any = True
    if hit_any:
        px[i] = pi_x
        py[i] = pi_y
        vx[i] = vi_x
        vy[i] = vi_y
        cuda.atomic.add(accum_jx, 0, -mass[i] * (vi_x - vi_x0))
        cuda.atomic.add(accum_jy, 0, -mass[i] * (vi_y - vi_y0))


# ── Device setup / detection ────────────────────────────────────────
def init_cuda() -> bool:
    """Patch CUDA 13.x DLL paths, probe the device, return GPU usability.

    CUDA 13.x relocated its shared libraries from ``bin/`` and ``nvvm/bin/`` to
    ``bin/x64/`` and ``nvvm/bin/x64/``; Numba's path helpers assume the old
    layout. We patch them (harmless on 12.x), prepend the DLL dirs, then
    probe-compile a trivial kernel to catch any remaining toolchain mismatch.
    """
    global CUDA_AVAILABLE
    try:
        cuda_base = os.environ.get("CUDA_PATH", "")
        if cuda_base:
            x64_nvvm = os.path.join(cuda_base, "nvvm", "bin", "x64")
            x64_bin = os.path.join(cuda_base, "bin", "x64")
            if os.path.isdir(x64_nvvm) and glob.glob(os.path.join(x64_nvvm, "nvvm64_*.dll")):
                try:
                    from numba.cuda import cuda_paths as _cp
                    _cp._nvvm_lib_dir = lambda: ("nvvm", "bin", "x64")
                    if os.path.isdir(x64_bin):
                        _cp._cudalib_path = lambda: os.path.join("bin", "x64")
                except Exception:
                    pass
            for sub in (r"nvvm\bin\x64", r"bin\x64", r"nvvm\bin", "bin"):
                p = os.path.join(cuda_base, sub)
                if os.path.isdir(p):
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                    try:
                        os.add_dll_directory(p)
                    except Exception:
                        pass

        cuda.gpus[0]  # raises if no supported device

        @cuda.jit
        def _probe(x):
            pass

        probe = cuda.device_array(32, dtype=np.float32)
        _probe[1, 32](probe)
        cuda.synchronize()
        CUDA_AVAILABLE = True
    except Exception:
        CUDA_AVAILABLE = False
    return CUDA_AVAILABLE
