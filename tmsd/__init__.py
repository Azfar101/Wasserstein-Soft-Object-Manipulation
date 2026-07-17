"""
Transport-Metric Skill Discovery (TMSD).

Unsupervised skill discovery whose diversity signal is a *physical*
optimal-transport metric on the external material state. Follows the
METRA/LSD distance-maximizing template — maximize E[(φ(g′) − φ(g))·z]
subject to ‖φ(g) − φ(g′)‖ ≤ d(g, g′) — but with the ground metric d
instantiated as the Wasserstein-2 distance between soil mass
distributions, and φ conditioned only on the external (soil) state.

Skills that merely wave the arm earn no reward; skills must differ by
how they transport soil mass.
"""

from .measures import soil_mass_histogram
from .metrics import GROUND_METRICS, w2_1d, euclidean
from .wrappers import SkillDiscoveryEnv
from .trainer import TMSDTrainer, TMSDConfig

__all__ = [
    "soil_mass_histogram",
    "GROUND_METRICS", "w2_1d", "euclidean",
    "SkillDiscoveryEnv",
    "TMSDTrainer", "TMSDConfig",
]
