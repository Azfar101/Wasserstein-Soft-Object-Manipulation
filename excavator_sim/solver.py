"""
Physics stepping: dispatches one frame to the CPU or CUDA backend.

The :class:`Solver` owns the spatial-hash buffers, the (lazy) GPU device arrays,
and the bucket-reaction accounting. ``step`` advances ``sub_steps × dt`` seconds.
The bucket inputs are bundled into a :class:`BucketState` so the per-frame call
site stays small instead of threading ~20 positional arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import DOMAIN_WIDTH, DOMAIN_HEIGHT, CELL_SIZE
from .config import SimConfig
from .spatial_hash import GRID_COLS, GRID_ROWS, GRID_TOTAL, MAX_PER_CELL, build_grid
from . import cpu_backend as cpu
from . import cuda_backend as gpu

_BOULDER_DAMP = 0.92  # per-frame velocity damping for boulders in contact


@dataclass
class BucketState:
    """Excavator bucket collider: precomputed wall segments + wrist velocity.

    Segment endpoints come from :func:`excavator_sim.geometry.bucket_segments`
    (constant across a frame's substeps, since the arm integrates once per frame).
    """
    active: bool = False
    seg_x0: Optional[np.ndarray] = None
    seg_y0: Optional[np.ndarray] = None
    seg_x1: Optional[np.ndarray] = None
    seg_y1: Optional[np.ndarray] = None
    vx: float = 0.0
    vy: float = 0.0


class Solver:
    """Per-frame physics stepping over a :class:`ParticleSystem`."""

    def __init__(self, config: SimConfig, force_cpu: bool = False):
        self._bury_threshold = config.boulder.bury_threshold

        if force_cpu or config.use_gpu is False:
            self.use_cuda = False
        else:
            self.use_cuda = gpu.init_cuda()

        # Spatial-hash buffers (host).
        self.cell_counts = np.zeros(GRID_TOTAL, dtype=np.int32)
        self.cell_particles = np.zeros((GRID_TOTAL, MAX_PER_CELL), dtype=np.int32)

        # Bucket reaction (Newton's 3rd law: −Σ particle impulses), read by the env.
        self.bucket_reaction_fx = 0.0
        self.bucket_reaction_fy = 0.0
        self.bucket_contact_count = 0
        self._bucket_jx = np.zeros(1, dtype=np.float64)
        self._bucket_jy = np.zeros(1, dtype=np.float64)

        if self.use_cuda:
            self._dev_capacity = 0
            self._dev = {}
            self._h_stage = np.empty(1, dtype=np.float32)
            self.d_cell_counts = gpu.cuda.device_array(GRID_TOTAL, dtype=np.int32)
            self.d_cell_particles = gpu.cuda.device_array(
                (GRID_TOTAL, MAX_PER_CELL), dtype=np.int32)
            self._d_seg = {k: gpu.cuda.device_array(5, dtype=np.float32)
                           for k in ("x0", "y0", "x1", "y1")}
            self._d_jx = gpu.cuda.device_array(1, dtype=np.float32)
            self._d_jy = gpu.cuda.device_array(1, dtype=np.float32)

    # ── public API ──────────────────────────────────────────────────
    def step(self, ps, cfg: SimConfig, *,
             bucket: Optional[BucketState] = None) -> None:
        """Advance the simulation by ``cfg.sub_steps × cfg.dt`` seconds."""
        n = ps.count
        self.bucket_reaction_fx = 0.0
        self.bucket_reaction_fy = 0.0
        self.bucket_contact_count = 0
        if n == 0:
            return

        bucket = bucket or BucketState()
        impl = self._step_cuda if self.use_cuda else self._step_cpu
        impl(ps, cfg, n, bucket)

        frame_dt = cfg.dt * cfg.sub_steps
        if frame_dt > 0.0:
            self.bucket_reaction_fx /= frame_dt
            self.bucket_reaction_fy /= frame_dt

    # ── shared helpers ──────────────────────────────────────────────
    def _boulder_stats(self, ps, n):
        """Return ``(boulder_indices, n_boulders, soil_r_max)``."""
        boul = np.where(ps.is_boulder[:n])[0].astype(np.int32)
        n_boul = len(boul)
        if n_boul > 0:
            soil = ~ps.is_boulder[:n]
            soil_r_max = float(ps.radius[:n][soil].max()) if soil.any() \
                else float(ps.radius[:n].max())
        else:
            soil_r_max = float(ps.radius[:n].max())
            boul = np.empty(0, dtype=np.int32)
        return boul, n_boul, soil_r_max

    # ── CPU path ────────────────────────────────────────────────────
    def _step_cpu(self, ps, cfg, n, bucket):
        boul, n_boul, soil_r_max = self._boulder_stats(ps, n)
        e_star = cfg.e_star
        beta = cfg.damping_beta
        soil = cfg.soil

        if bucket.active:
            if len(self._bucket_jx) < n:
                self._bucket_jx = np.zeros(n, dtype=np.float64)
                self._bucket_jy = np.zeros(n, dtype=np.float64)
            jx = self._bucket_jx[:n]
            jy = self._bucket_jy[:n]
            jx[:] = 0.0
            jy[:] = 0.0

        for _ in range(cfg.sub_steps):
            ps.ax[:n] = 0.0
            ps.ay[:n] = 0.0
            build_grid(ps.px, ps.py, n, self.cell_counts, self.cell_particles)
            cpu.solve_forces(
                ps.px, ps.py, ps.vx, ps.vy, ps.ax, ps.ay,
                ps.radius, ps.mass, ps.inv_mass, n,
                self.cell_counts, self.cell_particles,
                e_star, soil.friction, beta, soil.gravity,
                DOMAIN_WIDTH, DOMAIN_HEIGHT,
                soil.cohesion, soil.cohesion_strength, soil.rolling_resistance,
                soil_r_max, ps.is_boulder[:n], boul, n_boul, ps.friction[:n])

            if bucket.active:
                cpu.apply_bucket_scoop(
                    ps.px, ps.py, ps.vx, ps.vy,
                    ps.radius, ps.mass, ps.inv_mass, n,
                    bucket.seg_x0, bucket.seg_y0, bucket.seg_x1, bucket.seg_y1,
                    bucket.vx, bucket.vy, e_star, soil.friction, cfg.dt, jx, jy)

            saved = [(float(ps.px[b]), float(ps.py[b])) for b in boul]
            cpu.integrate(ps.px, ps.py, ps.vx, ps.vy, ps.ax, ps.ay, n, cfg.dt,
                          DOMAIN_WIDTH, DOMAIN_HEIGHT, ps.radius)

            if not bucket.active:
                self._lock_buried_boulders(ps, n, boul, saved)

        self._damp_contacting_boulders(ps, n, boul)

        if bucket.active:
            self.bucket_reaction_fx = -float(jx.sum())
            self.bucket_reaction_fy = -float(jy.sum())
            self.bucket_contact_count = int(np.count_nonzero(jx))

    def _lock_buried_boulders(self, ps, n, boul, saved):
        for idx, b in enumerate(boul):
            b = int(b)
            nn = cpu.count_boulder_neighbors(
                ps.px, ps.py, ps.radius, ps.is_boulder[:n], n, b, 0.05)
            if nn > self._bury_threshold:
                ps.px[b], ps.py[b] = saved[idx]
                ps.vx[b] = 0.0
                ps.vy[b] = 0.0

    def _damp_contacting_boulders(self, ps, n, boul):
        for b in boul:
            b = int(b)
            nn = cpu.count_boulder_neighbors(
                ps.px, ps.py, ps.radius, ps.is_boulder[:n], n, b, 0.05)
            if nn > 0:
                ps.vx[b] *= _BOULDER_DAMP
                ps.vy[b] *= _BOULDER_DAMP

    # ── CUDA path ───────────────────────────────────────────────────
    def _ensure_device(self, n):
        if n <= self._dev_capacity:
            return
        cap = max(n, 1024)
        self._dev_capacity = cap
        f32 = lambda: gpu.cuda.device_array(cap, dtype=np.float32)
        self._dev = {k: f32() for k in
                     ("px", "py", "vx", "vy", "ax", "ay",
                      "radius", "mass", "inv_mass", "friction")}
        self._dev["is_boulder"] = gpu.cuda.device_array(cap, dtype=np.bool_)
        self._dev["boul_indices"] = gpu.cuda.device_array(max(cap, 64), dtype=np.int32)
        self._h_stage = np.empty(cap, dtype=np.float32)

    def _h2d(self, key, src, n):
        np.copyto(self._h_stage[:n], src[:n])           # float64 → float32
        self._dev[key][:n].copy_to_device(self._h_stage[:n])

    def _d2h(self, dst, key, n):
        self._dev[key][:n].copy_to_host(self._h_stage[:n])
        np.copyto(dst[:n], self._h_stage[:n])           # float32 → float64

    def _step_cuda(self, ps, cfg, n, bucket):
        self._ensure_device(n)
        d = self._dev
        boul, n_boul, soil_r_max = self._boulder_stats(ps, n)
        soil = cfg.soil
        e_star = np.float32(cfg.e_star)
        beta = np.float32(cfg.damping_beta)

        # Boulder buriedness on host (before transfer); restored after.
        saved = []
        for b in boul:
            b = int(b)
            buried = (not bucket.active) and cpu.count_boulder_neighbors(
                ps.px, ps.py, ps.radius, ps.is_boulder[:n], n, b, 0.05) > self._bury_threshold
            saved.append((float(ps.px[b]), float(ps.py[b]), buried))

        for key in ("px", "py", "vx", "vy", "radius", "mass", "inv_mass", "friction"):
            self._h2d(key, getattr(ps, key), n)
        d["is_boulder"][:n].copy_to_device(ps.is_boulder[:n])
        if n_boul > 0:
            d["boul_indices"][:n_boul].copy_to_device(boul)

        threads = 256
        blocks = (n + threads - 1) // threads
        gblocks = (GRID_TOTAL + threads - 1) // threads

        gravity = np.float32(soil.gravity)
        friction = np.float32(soil.friction)
        cohesion_s = np.float32(soil.cohesion_strength)
        rolling = np.float32(soil.rolling_resistance)
        domain_w = np.float32(DOMAIN_WIDTH)
        domain_h = np.float32(DOMAIN_HEIGHT)
        cell_size = np.float32(CELL_SIZE)
        dt = np.float32(cfg.dt)
        soil_r_max = np.float32(soil_r_max)

        if bucket.active:
            for k, arr in (("x0", bucket.seg_x0), ("y0", bucket.seg_y0),
                           ("x1", bucket.seg_x1), ("y1", bucket.seg_y1)):
                self._d_seg[k].copy_to_device(arr.astype(np.float32))
            gpu.zero_f32_kernel[1, 1](self._d_jx, 1)
            gpu.zero_f32_kernel[1, 1](self._d_jy, 1)
            wrist_vx = np.float32(bucket.vx)
            wrist_vy = np.float32(bucket.vy)

        for _ in range(cfg.sub_steps):
            gpu.zero_f32_kernel[blocks, threads](d["ax"], n)
            gpu.zero_f32_kernel[blocks, threads](d["ay"], n)
            gpu.zero_i32_kernel[gblocks, threads](self.d_cell_counts, GRID_TOTAL)
            gpu.build_grid_kernel[blocks, threads](
                d["px"], d["py"], n, self.d_cell_counts, self.d_cell_particles,
                GRID_COLS, GRID_ROWS, cell_size, MAX_PER_CELL, d["is_boulder"])
            gpu.solve_kernel[blocks, threads](
                d["px"], d["py"], d["vx"], d["vy"], d["ax"], d["ay"],
                d["radius"], d["mass"], d["inv_mass"], n,
                self.d_cell_counts, self.d_cell_particles,
                e_star, friction, beta, gravity, domain_w, domain_h,
                soil.cohesion, cohesion_s, rolling,
                GRID_COLS, GRID_ROWS, cell_size, MAX_PER_CELL,
                soil_r_max, d["boul_indices"], n_boul, d["friction"])

            if bucket.active:
                gpu.bucket_scoop_kernel[blocks, threads](
                    d["px"], d["py"], d["vx"], d["vy"],
                    d["radius"], d["mass"], d["inv_mass"], n,
                    self._d_seg["x0"], self._d_seg["y0"],
                    self._d_seg["x1"], self._d_seg["y1"],
                    wrist_vx, wrist_vy, e_star, friction, dt,
                    self._d_jx, self._d_jy)

            gpu.integrate_kernel[blocks, threads](
                d["px"], d["py"], d["vx"], d["vy"], d["ax"], d["ay"], n, dt,
                domain_w, domain_h, d["radius"])

        for key in ("px", "py", "vx", "vy"):
            self._d2h(getattr(ps, key), key, n)

        if bucket.active:
            hx = np.zeros(1, dtype=np.float32)
            hy = np.zeros(1, dtype=np.float32)
            self._d_jx.copy_to_host(hx)
            self._d_jy.copy_to_host(hy)
            self.bucket_reaction_fx = float(hx[0])
            self.bucket_reaction_fy = float(hy[0])
            self.bucket_contact_count = 1 if (hx[0] or hy[0]) else 0

        # Boulder handling on the host, matching the CPU path.
        self._damp_contacting_boulders(ps, n, boul)
        # Restore buried boulders on the host after D2H.
        for idx, b in enumerate(boul):
            b = int(b)
            sx, sy, buried = saved[idx]
            if buried:
                ps.px[b], ps.py[b] = sx, sy
                ps.vx[b] = 0.0
                ps.vy[b] = 0.0
