"""
Particle storage in Structure-of-Arrays (SoA) layout.

All per-particle state lives in contiguous NumPy arrays for cache-friendly
access and cheap GPU transfer. The set of arrays is declared once in
``_ARRAYS`` so that adding a field stays a one-line change and the add / remove
/ compaction paths never drift out of sync.

Spawning helpers (:func:`spawn_pile`, :func:`place_boulder`) take an explicit
NumPy ``Generator`` so episodes are reproducible under a seed.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig


class ParticleSystem:
    """SoA container for all particle state (no global RNG, no sleeping)."""

    # Every per-particle array, declared once. ``is_boulder`` is boolean; the
    # rest are float64. add / remove / _compact all iterate this tuple.
    _ARRAYS = (
        "px", "py", "vx", "vy", "ax", "ay",
        "radius", "mass", "inv_mass", "friction", "is_boulder",
    )

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.count = 0
        for name in self._ARRAYS:
            dtype = np.bool_ if name == "is_boulder" else np.float64
            setattr(self, name, np.zeros(self.capacity, dtype=dtype))

    # ── mutation ────────────────────────────────────────────────────
    def add(self, x: float, y: float, radius: float, *,
            vx: float = 0.0, vy: float = 0.0,
            density: float = 2650.0, friction: float = 0.6,
            is_boulder: bool = False) -> int:
        """Add one particle. Returns its index, or -1 if the buffer is full."""
        if self.count >= self.capacity:
            return -1
        i = self.count
        self.px[i] = x
        self.py[i] = y
        self.vx[i] = vx
        self.vy[i] = vy
        self.ax[i] = 0.0
        self.ay[i] = 0.0
        self.radius[i] = radius
        m = density * np.pi * radius * radius
        self.mass[i] = m
        self.inv_mass[i] = 1.0 / m
        self.friction[i] = friction
        self.is_boulder[i] = is_boulder
        self.count += 1
        return i

    def add_batch(self, xs, ys, radii, *,
                  density: float = 2650.0, friction: float = 0.6,
                  is_boulder: bool = False) -> int:
        """Add many particles at once (vectorised). Returns the number added."""
        n = len(xs)
        free = self.capacity - self.count
        if n > free:
            n = free
        if n <= 0:
            return 0
        s, e = self.count, self.count + n
        xs = np.asarray(xs, dtype=np.float64)[:n]
        ys = np.asarray(ys, dtype=np.float64)[:n]
        radii = np.asarray(radii, dtype=np.float64)[:n]
        self.px[s:e] = xs
        self.py[s:e] = ys
        self.vx[s:e] = 0.0
        self.vy[s:e] = 0.0
        self.ax[s:e] = 0.0
        self.ay[s:e] = 0.0
        self.radius[s:e] = radii
        masses = density * np.pi * radii * radii
        self.mass[s:e] = masses
        self.inv_mass[s:e] = 1.0 / masses
        self.friction[s:e] = friction
        self.is_boulder[s:e] = is_boulder
        self.count = e
        return n

    def remove(self, index: int) -> None:
        """Remove one particle by swapping the last element into its slot."""
        if index < 0 or index >= self.count:
            return
        last = self.count - 1
        if index != last:
            for name in self._ARRAYS:
                arr = getattr(self, name)
                arr[index] = arr[last]
        self.count -= 1

    def remove_near(self, x: float, y: float, remove_radius: float) -> int:
        """Remove every (non-boulder) particle within *remove_radius* of (x, y).

        Returns the number removed. Boulders are indestructible.
        """
        n = self.count
        dx = self.px[:n] - x
        dy = self.py[:n] - y
        dist_sq = dx * dx + dy * dy
        keep = (dist_sq > remove_radius * remove_radius) | self.is_boulder[:n]
        return self._compact(keep)

    def remove_boulders(self) -> int:
        """Remove all boulder particles. Returns the number removed."""
        keep = ~self.is_boulder[:self.count]
        return self._compact(keep)

    def clear(self) -> None:
        self.count = 0

    def _compact(self, keep) -> int:
        """Keep only particles where *keep* is True; compact arrays in place."""
        n = self.count
        new_count = int(np.count_nonzero(keep))
        if new_count == n:
            return 0
        for name in self._ARRAYS:
            arr = getattr(self, name)
            arr[:new_count] = arr[:n][keep]
        self.count = new_count
        return n - new_count

    def keep_where(self, keep) -> int:
        """Keep only particles where *keep* (bool, length ``count``) is True.

        Returns the number removed. Boulders are preserved regardless.
        """
        keep = np.asarray(keep, dtype=bool) | self.is_boulder[:self.count]
        return self._compact(keep)

    # ── views ───────────────────────────────────────────────────────
    @property
    def soil_mask(self):
        """Boolean mask of live non-boulder particles."""
        return ~self.is_boulder[:self.count]


# ── Population helpers (reproducible under a seeded Generator) ───────
def spawn_pile(ps: ParticleSystem, cfg: SimConfig, rng: np.random.Generator) -> int:
    """Lay down a packed hexagonal soil bed in the bottom of the domain.

    Returns the number of grains placed (capped by ``cfg.spawn.target_count``
    and the particle buffer capacity).
    """
    sp = cfg.spawn
    pp = cfg.particle
    spacing = pp.radius_max * 2.2
    ground_h = cfg.domain_height * sp.pile_height_frac
    margin = pp.radius_max * 1.5

    rows = max(0, int((ground_h - margin) / (spacing * 0.866)))
    xs, ys = [], []
    for row in range(rows):
        x_off = (spacing * 0.5) if (row % 2) else 0.0
        y = margin + row * spacing * 0.866
        x = margin + x_off
        while x <= cfg.domain_width - margin:
            xs.append(x)
            ys.append(y)
            x += spacing
            if len(xs) >= sp.target_count:
                break
        if len(xs) >= sp.target_count:
            break

    if not xs:
        return 0
    radii = rng.uniform(pp.radius_min, pp.radius_max, size=len(xs))
    return ps.add_batch(np.array(xs), np.array(ys), radii,
                        density=pp.density, friction=cfg.soil.friction)


def place_boulder(ps: ParticleSystem, cfg: SimConfig, x: float, y: float) -> int:
    """Place a single indestructible boulder, clamped inside the domain."""
    r = cfg.boulder.radius
    bx = float(np.clip(x, r, cfg.domain_width - r))
    by = float(np.clip(y, r, cfg.domain_height - r))
    return ps.add(bx, by, r, density=cfg.boulder.density,
                  friction=cfg.boulder.friction, is_boulder=True)
