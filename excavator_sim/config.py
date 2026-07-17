"""
Runtime-tunable configuration for the excavator simulation.

Everything here can vary per :class:`ExcavatorEnv` instance without recompiling
the physics kernels — soil material properties, arm hydraulics and geometry,
the bucket, particle spawning, rendering, and the default task. Values that are
baked into the Numba kernels instead live in :mod:`excavator_sim.constants`.

The config is a tree of small mutable dataclasses so experiments read naturally::

    cfg = SimConfig()
    cfg.soil.friction = 0.7
    cfg.arm.stall_torque = (250_000, 180_000, 55_000)
    env = ExcavatorEnv(cfg)

Derived quantities (``e_star``, ``damping_beta``, ``pixels_per_meter``) are
computed on demand so they always reflect the current field values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

from .constants import DOMAIN_WIDTH, DOMAIN_HEIGHT, POISSON_RATIO


# ── Soil / contact material ─────────────────────────────────────────
@dataclass
class SoilParams:
    """Bulk soil material properties (passed to the contact solver)."""
    gravity: float = 9.81             # m/s²
    friction: float = 0.6             # Coulomb μ for soil–soil contacts
    restitution: float = 0.001        # ~inelastic; drives contact damping
    rolling_resistance: float = 0.6   # resists rolling when supported
    youngs_modulus: float = 220.0e6   # Pa, soft-sphere contact stiffness
    cohesion: bool = True             # wet-soil attraction within a cutoff
    cohesion_strength: float = 200.0  # N/m


# ── Particles ───────────────────────────────────────────────────────
@dataclass
class ParticleParams:
    """Grain size/mass distribution and capacity."""
    radius_min: float = 0.03          # m
    radius_max: float = 0.06          # m
    density: float = 2650.0           # kg/m² (2D areal; quartz sand)
    max_particles: int = 15000


# ── Boulders ────────────────────────────────────────────────────────
@dataclass
class BoulderParams:
    """Indestructible rock obstacles placed in the soil."""
    radius: float = 0.35              # m (~35 cm)
    density: float = 2700.0           # kg/m² (granite)
    friction: float = 0.7             # Coulomb μ for rock–soil contacts
    bury_threshold: int = 12          # soil neighbours above which it locks


# ── Excavator arm (velocity-governed hydraulics) ────────────────────
@dataclass
class ArmParams:
    """Two-link arm geometry, link masses, and hydraulic actuator limits.

    The per-joint 3-tuples are ordered ``(shoulder, elbow, wrist)`` =
    ``(boom, stick, bucket)``.
    """
    pivot_x: float = 2.00             # boom-root world x (m)
    pivot_y: float = 3.60             # boom-root world y (m)
    boom_length: float = 4.2          # m
    stick_length: float = 3.2         # m
    boom_mass: float = 1200.0         # kg
    stick_mass: float = 600.0         # kg
    bucket_mass: float = 400.0        # kg (empty)

    # Hydraulic flow limits → max angular velocity per joint (rad/s)
    max_omega: Tuple[float, float, float] = (0.4, 0.6, 1.0)
    # Spool response → angular acceleration limit per joint (rad/s²)
    accel: Tuple[float, float, float] = (1.5, 2.0, 4.0)
    # Relief-valve pressure → contact torque that fully stalls a joint (N·m)
    stall_torque: Tuple[float, float, float] = (300_000.0, 200_000.0, 60_000.0)

    velocity_gain: float = 4.0        # position-error → desired-velocity gain
    wrist_rate: float = 1.25          # bucket rotation rate from input (rad/s)

    # Joint angle limits (rad): (shoulder, elbow, wrist)
    limits: Tuple[Tuple[float, float], ...] = (
        (-math.pi * 0.5, math.pi * 0.5),
        (0.2, math.pi * 0.95),
        (-math.pi * 0.95, math.pi * 0.3),
    )

    # Heavier payload mildly throttles max speed (0 disables; ~realistic).
    payload_speed_droop: float = 0.0  # fractional droop per 1000 kg


# ── Bucket collider ─────────────────────────────────────────────────
@dataclass
class BucketParams:
    radius: float = 1.00              # bucket scale (m)


# ── Particle spawning ───────────────────────────────────────────────
@dataclass
class SpawnParams:
    """Initial soil pile laid down at reset."""
    pile_height_frac: float = 0.45    # fraction of domain height to fill
    target_count: int = 1200          # cap on grains placed (deeper = heavier)


# ── Rendering ───────────────────────────────────────────────────────
@dataclass
class RenderParams:
    window_width: int = 1280
    window_height: int = 720
    fps_cap: int = 60
    bg_color: Tuple[int, int, int] = (30, 30, 35)
    wall_color: Tuple[int, int, int] = (100, 100, 110)
    ui_color: Tuple[int, int, int] = (220, 220, 220)
    color_small: Tuple[int, int, int] = (220, 190, 130)
    color_large: Tuple[int, int, int] = (160, 120, 70)
    color_boulder: Tuple[int, int, int] = (110, 110, 125)
    # Torque-gauge full-scale per joint (N·m): (shoulder, elbow, wrist)
    torque_gauge_max: Tuple[float, float, float] = (250_000.0, 120_000.0, 40_000.0)


# ── Default excavation task ─────────────────────────────────────────
@dataclass
class TaskParams:
    """Weights/thresholds for the built-in :class:`ExcavationTask`."""
    lift_goal_y: float = 4.3          # wrist height (m) defining a "lifted" scoop
    bucket_capacity_kg: float = 400.0  # payload is capped here (anti reward-hack:
                                       # a buried bucket reads inflated geometric mass)
    reward_lift: float = 1.0          # per kg of (capped) payload newly above goal
    reward_capture: float = 0.05      # per kg of (capped) payload newly in the bowl
    penalty_energy: float = 0.002     # per unit of actuator effort
    penalty_time: float = 0.01        # per step
    success_hold_steps: int = 30      # steps of held lift to count as success
    success_payload_kg: float = 100.0  # lifted payload required for success


# ── Top-level config ────────────────────────────────────────────────
@dataclass
class SimConfig:
    """Complete environment configuration."""
    soil: SoilParams = field(default_factory=SoilParams)
    particle: ParticleParams = field(default_factory=ParticleParams)
    boulder: BoulderParams = field(default_factory=BoulderParams)
    arm: ArmParams = field(default_factory=ArmParams)
    bucket: BucketParams = field(default_factory=BucketParams)
    spawn: SpawnParams = field(default_factory=SpawnParams)
    render: RenderParams = field(default_factory=RenderParams)
    task: TaskParams = field(default_factory=TaskParams)

    # Time integration
    dt: float = 0.0005                # fixed timestep (s)
    sub_steps: int = 30               # physics sub-steps per env step

    # Backend / control
    use_gpu: Optional[bool] = None    # None = auto-detect; False = force CPU
    control_mode: str = "velocity"    # "velocity" | "position" (action meaning)

    # Observation
    heightfield_bins: int = 16        # terrain-elevation samples across width

    # Domain mirrors (read-only convenience; the grid is sized from constants)
    domain_width: float = DOMAIN_WIDTH
    domain_height: float = DOMAIN_HEIGHT

    # ── Derived quantities ──────────────────────────────────────────
    @property
    def e_star(self) -> float:
        """Effective Young's modulus for equal-material contacts."""
        return self.soil.youngs_modulus / (2.0 * (1.0 - POISSON_RATIO ** 2))

    @property
    def damping_beta(self) -> float:
        """Normal-damping coefficient derived from the restitution."""
        ln_e = math.log(max(self.soil.restitution, 1e-6))
        return -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)

    @property
    def pixels_per_meter(self) -> float:
        return self.render.window_width / self.domain_width

    def copy(self) -> "SimConfig":
        """Deep-ish copy (each dataclass branch is replaced fresh)."""
        return replace(
            self,
            soil=replace(self.soil),
            particle=replace(self.particle),
            boulder=replace(self.boulder),
            arm=replace(self.arm),
            bucket=replace(self.bucket),
            spawn=replace(self.spawn),
            render=replace(self.render),
            task=replace(self.task),
        )


# ── Named soil presets ──────────────────────────────────────────────
@dataclass
class SoilPreset:
    """A named bundle of soil + density overrides applied onto a SimConfig."""
    name: str
    gravity: float
    friction: float
    restitution: float
    rolling_resistance: float
    youngs_modulus: float
    cohesion: bool
    cohesion_strength: float
    density: float

    def apply(self, cfg: SimConfig) -> SimConfig:
        cfg.soil.gravity = self.gravity
        cfg.soil.friction = self.friction
        cfg.soil.restitution = self.restitution
        cfg.soil.rolling_resistance = self.rolling_resistance
        cfg.soil.youngs_modulus = self.youngs_modulus
        cfg.soil.cohesion = self.cohesion
        cfg.soil.cohesion_strength = self.cohesion_strength
        cfg.particle.density = self.density
        return cfg


SOIL_PRESETS: Tuple[SoilPreset, ...] = (
    SoilPreset("Default",       9.81, 0.60, 0.001, 0.60, 220.0e6, True,  200.0, 2650.0),
    SoilPreset("Dry Sand",      9.81, 0.55, 0.05,  0.30, 1.0e7,   False,   0.0, 2650.0),
    SoilPreset("Wet Sand",      9.81, 0.65, 0.01,  0.50, 5.0e7,   True,  120.0, 2650.0),
    SoilPreset("Gravel",        9.81, 0.70, 0.15,  0.20, 2.0e8,   False,   0.0, 2800.0),
    SoilPreset("Hard Clay",     9.81, 0.45, 0.001, 0.80, 2.2e8,   True,  300.0, 2100.0),
    SoilPreset("Wet Clay",      9.81, 0.35, 0.001, 0.60, 1.0e8,   True,  200.0, 2000.0),
    SoilPreset("Loose Topsoil", 9.81, 0.40, 0.02,  0.15, 5.0e6,   True,   40.0, 1500.0),
    SoilPreset("Moon Regolith", 1.62, 0.80, 0.05,  0.70, 8.0e7,   True,   10.0, 1800.0),
)


def preset_config(name: str) -> SimConfig:
    """Return a fresh :class:`SimConfig` with the named soil preset applied."""
    cfg = SimConfig()
    for preset in SOIL_PRESETS:
        if preset.name.lower() == name.lower():
            return preset.apply(cfg)
    raise KeyError(f"unknown soil preset {name!r}; "
                   f"options: {[p.name for p in SOIL_PRESETS]}")
