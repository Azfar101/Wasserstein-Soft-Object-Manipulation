"""
Task definitions: the reward / termination / info layer.

A :class:`Task` is deliberately thin so you can drop in your own objective
without touching the environment. Subclass it (or pass ``reward_fn=`` to the
env) and override any of :meth:`reset`, :meth:`reward`, :meth:`terminated`,
:meth:`info`. The env passes itself in, so a task can read everything it needs:
``env.payload_mass``, ``env.arm_state``, ``env.contact_torques``,
``env.boulder``, ``env.ps``, ``env.cfg``, ``env.steps``.

The default :class:`ExcavationTask` rewards *increases* in captured and lifted
soil (a delta/potential formulation, so idling with a full bucket earns nothing),
minus small effort and time penalties.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import TaskParams


class Task:
    """Base task: no reward, never terminates. Override what you need."""

    def reset(self, env) -> None:
        pass

    def reward(self, env, action) -> float:
        return 0.0

    def terminated(self, env) -> bool:
        return False

    def info(self, env) -> Dict[str, object]:
        return {}


class ExcavationTask(Task):
    """Scoop soil into the bucket and lift it above a goal height."""

    def __init__(self, params: TaskParams):
        self.p = params
        self._prev_payload = 0.0
        self._prev_lifted = 0.0
        self._hold = 0

    def reset(self, env) -> None:
        self._prev_payload = self._captured(env)
        self._prev_lifted = self._lifted(env)
        self._hold = 0

    def _captured(self, env) -> float:
        """Payload in the bowl, capped at the realistic bucket capacity."""
        return min(env.payload_mass, self.p.bucket_capacity_kg)

    def _lifted(self, env) -> float:
        """Captured mass currently held above the goal height (else 0)."""
        return self._captured(env) if env.arm_state["wrist_y"] > self.p.lift_goal_y else 0.0

    def reward(self, env, action) -> float:
        p = self.p
        payload = self._captured(env)
        lifted = self._lifted(env)

        r = p.reward_capture * max(0.0, payload - self._prev_payload)
        r += p.reward_lift * max(0.0, lifted - self._prev_lifted)
        r -= p.penalty_energy * float(np.sum(np.abs(action)))
        r -= p.penalty_time

        self._prev_payload = payload
        self._prev_lifted = lifted
        self._hold = self._hold + 1 if lifted >= p.success_payload_kg else 0
        return r

    def terminated(self, env) -> bool:
        return self._hold >= self.p.success_hold_steps

    def info(self, env) -> Dict[str, object]:
        return {
            "lifted_kg": self._lifted(env),
            "hold_steps": self._hold,
            "is_success": self._hold >= self.p.success_hold_steps,
        }


class RewardFnTask(ExcavationTask):
    """Excavation termination/info, but reward delegated to a user callable.

    The callable receives ``(env, action)`` and returns a float. Convenience for
    the env's ``reward_fn=`` argument; subclass :class:`Task` for full control.
    """

    def __init__(self, params: TaskParams, reward_fn):
        super().__init__(params)
        self._fn = reward_fn

    def reward(self, env, action) -> float:
        # Keep the success bookkeeping running for terminated()/info().
        super().reward(env, action)
        return float(self._fn(env, action))
