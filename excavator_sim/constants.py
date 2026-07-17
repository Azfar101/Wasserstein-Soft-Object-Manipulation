"""
Compile-time simulation constants.

These values are *baked into the Numba CPU/CUDA kernels* at JIT-compile time
(the kernels close over them as module globals), so they cannot vary per
environment instance without recompiling. They define the fixed "physical
fabric" of the world: domain size, the spatial-hash grid, and a few numerical
contact constants that are not exposed as runtime knobs.

Everything an RL environment is likely to want to vary at runtime — soil
material properties, arm strength, the task — lives in :mod:`excavator_sim.config`
as a :class:`~excavator_sim.config.SimConfig` and is threaded through the
kernels as *arguments*, not baked here.

See ``config.py`` for the runtime-tunable parameters.
"""

# ── Simulation domain (metres) ──────────────────────────────────────
# The simulation runs in SI units; rendering maps these to pixels.
DOMAIN_WIDTH = 12.8   # metres
DOMAIN_HEIGHT = 7.2   # metres

# ── Spatial-hash grid ───────────────────────────────────────────────
# Cell side length. Must be at least ~2× the largest particle radius so the
# adaptive neighbour search (search_r in the solver) stays cheap. Sized for
# the default particle radius range (max 0.06 m → 2.5× headroom).
CELL_SIZE = 0.15

# ── Hertz-Mindlin numerics (kernel-baked) ───────────────────────────
POISSON_RATIO = 0.3
# Maximum overlap used in force calculation, as a fraction of the effective
# radius R*. Clamps force when particles deeply interpenetrate.
MAX_OVERLAP_RATIO = 0.30

# ── Integration safety (kernel-baked) ───────────────────────────────
# Per-substep velocity damping (mild air drag for airborne grains).
GLOBAL_DAMPING = 0.999
# Hard velocity cap (m/s) preventing numerical explosions.
MAX_VELOCITY = 15.0

# ── Excavator bucket profile (single source of truth) ───────────────
# The bucket bowl is a polyline of 5 points P0..P4 measured from the wrist pin
# in the (mouth, cavity) frame, each coefficient scaled by the bucket radius:
#       P_k = wrist + r * (along_k * mouth_axis + depth_k * cavity_axis)
# P0 is the pin itself (0, 0). The remaining four points are below.
# Used identically by the CPU collider, the CUDA collider, and the renderer.
BUCKET_PROFILE = (
    (0.00, 0.80),   # P1 back plate
    (0.40, 0.95),   # P2 bottom-back
    (0.75, 0.65),   # P3 bottom-front
    (0.95, 0.10),   # P4 teeth / lip
)
# Short stick-tip connector behind the wrist, as a fraction of bucket radius.
BUCKET_LINK_FRAC = 0.48

# ── Bucket collider tuning (kernel-baked) ───────────────────────────
BUCKET_WALL_T = 0.03       # collision half-thickness of each bucket wall (m)
BUCKET_RESTITUTION = 0.05  # bucket-wall bounce (nearly inelastic)
BUCKET_BAUMGARTE = 1.00    # positional penetration correction factor
