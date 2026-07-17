"""excavator_sim — a 2D DEM soil + excavator-arm simulation and RL environment.

Quick start
-----------
>>> import excavator_sim as xs
>>> env = xs.ExcavatorEnv(render_mode=None)          # or xs.make_env("Wet Sand")
>>> obs, info = env.reset(seed=0)
>>> obs, reward, term, trunc, info = env.step(env.action_space.sample())

Or via Gymnasium's registry:
>>> import gymnasium, excavator_sim
>>> env = gymnasium.make("Excavator-v0")

The physics layer (``ParticleSystem``, ``Solver``, ``ArmDynamics``) is usable
on its own without Gymnasium if you only want the simulator.
"""

from .config import (
    SimConfig, SoilParams, ParticleParams, BoulderParams, ArmParams,
    BucketParams, SpawnParams, RenderParams, TaskParams,
    SoilPreset, SOIL_PRESETS, preset_config,
)
from .particles import ParticleSystem, spawn_pile, place_boulder
from .solver import Solver, BucketState
from .arm import ArmDynamics
from .tasks import Task, ExcavationTask, RewardFnTask
from .env import ExcavatorEnv

__all__ = [
    "ExcavatorEnv", "make_env",
    "SimConfig", "SoilParams", "ParticleParams", "BoulderParams", "ArmParams",
    "BucketParams", "SpawnParams", "RenderParams", "TaskParams",
    "SoilPreset", "SOIL_PRESETS", "preset_config",
    "ParticleSystem", "spawn_pile", "place_boulder",
    "Solver", "BucketState", "ArmDynamics",
    "Task", "ExcavationTask", "RewardFnTask",
]

__version__ = "1.0.0"


def make_env(preset: str = None, *, render_mode: str = None, **kwargs) -> ExcavatorEnv:
    """Convenience constructor: optional soil preset name + env kwargs."""
    cfg = preset_config(preset) if preset else SimConfig()
    return ExcavatorEnv(cfg, render_mode=render_mode, **kwargs)


# Register with Gymnasium (self-truncating, so no extra TimeLimit wrapper).
try:
    from gymnasium.envs.registration import register

    register(id="Excavator-v0", entry_point="excavator_sim.env:ExcavatorEnv",
             max_episode_steps=None)
except Exception:  # pragma: no cover - registration is best-effort
    pass
