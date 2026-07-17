"""
Interactive soil-shaping demo.

Two modes (TAB switches):

SHAPE mode — record & restore:
    left-drag    push soil with a broom disc
    right-drag   remove soil near cursor
    hold A       add soil at cursor
    SPACE        record current soil state as the GOAL (red line appears)
    ENTER        excavator autonomously restores the soil to the goal
    ESC          abort the excavator while it works

ZONES mode — excavate & dump:
    left-drag    mark the DIG zone (red band)
    right-drag   mark the DUMP zone (green band)
    C            clear zones
    ENTER        excavator excavates the dig zone into the dump zone
    ESC          abort while it works

Both modes:
    R            new random terrain (clears goal/zones)
    TAB          switch mode
    close window quit

The excavator uses only unsupervised skills + zero-shot latent steering
(z = phi(goal) - phi(now)); there is no reward function and no planner.
Steering stops on convergence, plateau (skill-granularity floor), or ESC.

Usage:
    python scripts/interactive_demo.py [--run-name tmsd_w2_rt_d4]
    python scripts/interactive_demo.py --selftest    # headless smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from excavator_sim.arm import ArmDynamics
from excavator_sim.renderer import Renderer
from tmsd.gc_trainer import GCTrainer, GCConfig
from tmsd.metrics import w2_1d_exact
from tmsd.planner import SkillMPC
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

ROOT = Path(__file__).resolve().parents[1]

BRUSH_R = 0.55          # meters, soil tools
PUSH_STRENGTH = 4.0     # m/s imparted by the broom
ADD_PER_FRAME = 3       # particles per frame while holding A
MPC_CANDIDATES = 10     # skills tried in imagination per decision
MPC_HORIZON = 90        # steps per imagined/executed skill segment
SEGMENT_SETTLE = 25     # physics ticks after re-homing the arm
MAX_SEGMENTS = 10
CONVERGED_W2 = 0.06     # near the ambient-settling floor
DIG_COLOR = (235, 90, 70)
DUMP_COLOR = (90, 200, 110)
WORKSPACE = (4.5, 9.5)  # arm's effective reach on the bed (m)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, default="tmsd_w2_rt_d4")
    p.add_argument("--ckpt", type=str, default="ckpt_latest.pt")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--selftest", action="store_true",
                   help="headless smoke test (dummy video driver)")
    return p.parse_args()


class App:
    def __init__(self, args):
        ckpt_path = ROOT / "runs" / args.run_name / args.ckpt
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.is_gc = saved.get("kind") == "gc"

        # render_mode=None: we drive our own renderer for custom HUD/overlays.
        hist_bins = 64 if self.is_gc else saved["cfg"].hist_bins
        self.env = SkillDiscoveryEnv(hist_bins=hist_bins,
                                     max_episode_steps=10 ** 9,
                                     randomize_terrain=True,
                                     render_mode=None)
        if self.is_gc:
            self.trainer = GCTrainer(saved["cfg"], self.env.grid_eval)
            self.trainer.load(str(ckpt_path))
            self.mpc = None
        else:
            cfg: TMSDConfig = saved["cfg"]
            assert getattr(cfg, "phi_input", "hist") == "hist"
            self.trainer = TMSDTrainer(cfg, self.env.grid)
            self.trainer.load(str(ckpt_path))
            self.mpc = SkillMPC(self.env, self.trainer,
                                n_candidates=MPC_CANDIDATES,
                                horizon=MPC_HORIZON)
        self.dev = self.trainer.device

        self.renderer = Renderer(self.env.env.cfg, mode="human")
        self.pg = self.renderer.pygame
        self.pg.display.set_caption(
            "TMSD interactive soil shaping"
            + ("  [goal-conditioned controller]" if self.is_gc else "  [skill-MPC]"))

        self.rng = np.random.default_rng()
        self.mode = "SHAPE"             # SHAPE | ZONES
        self.goal: np.ndarray | None = None
        self.dig: list | None = None    # [x0, x1]
        self.dump: list | None = None
        self.status = "sculpt the soil, SPACE to record goal"
        self.obs = None
        self.reset_terrain(args.seed)

    # ── terrain / state ──────────────────────────────────────────────
    def reset_terrain(self, seed=None):
        seed = int(self.rng.integers(0, 2 ** 31)) if seed is None else seed
        self.obs, _, _ = self.env.reset(seed=seed)
        self.goal = None
        self.dig = self.dump = None
        self.renderer.overlay_polyline = None
        self.renderer.overlay_zones = []

    def mouse_world(self):
        mx, my = self.pg.mouse.get_pos()
        cfg = self.env.env.cfg
        return (mx / self.renderer.ppm,
                cfg.domain_height - my / self.renderer.ppm)

    # ── soil editing tools ───────────────────────────────────────────
    def push_soil(self, wx, wy):
        ps = self.env.env.ps
        n = ps.count
        soil = ps.soil_mask
        dx = ps.px[:n] - wx
        dy = ps.py[:n] - wy
        d2 = dx * dx + dy * dy
        hit = (d2 < BRUSH_R * BRUSH_R) & soil
        if hit.any():
            d = np.sqrt(d2[hit]) + 1e-6
            ps.vx[:n][hit] += PUSH_STRENGTH * dx[hit] / d
            ps.vy[:n][hit] += PUSH_STRENGTH * 0.5 * np.abs(dy[hit]) / d

    def remove_soil(self, wx, wy):
        self.env.env.ps.remove_near(wx, wy, BRUSH_R * 0.7)

    def add_soil(self, wx, wy):
        ps, cfg = self.env.env.ps, self.env.env.cfg
        pp = cfg.particle
        for _ in range(ADD_PER_FRAME):
            r = self.rng.uniform(pp.radius_min, pp.radius_max)
            ox, oy = self.rng.uniform(-0.25, 0.25, size=2)
            ps.add(float(np.clip(wx + ox, r, cfg.domain_width - r)),
                   float(np.clip(wy + abs(oy), r, cfg.domain_height - r)),
                   r, density=pp.density, friction=cfg.soil.friction)

    # ── overlays / drawing ───────────────────────────────────────────
    def refresh_overlays(self):
        zones = [(WORKSPACE[0], WORKSPACE[1], (120, 150, 220), "reachable")]
        if self.dig:
            zones.append((min(self.dig), max(self.dig), DIG_COLOR, "DIG"))
        if self.dump:
            zones.append((min(self.dump), max(self.dump), DUMP_COLOR, "DUMP"))
        self.renderer.overlay_zones = zones
        if self.goal is not None:
            heights = self.env.hist_to_heights(self.goal)
            self.renderer.overlay_polyline = list(zip(self.env.grid, heights))
            self.renderer.overlay_label = "goal profile"
        else:
            self.renderer.overlay_polyline = None

    def draw(self, extra=()):
        e = self.env.env
        lines = [f"[{self.mode}]  {self.status}", *extra,
                 "TAB mode | R new terrain | ENTER go | ESC stop"]
        if self.mode == "SHAPE":
            lines.insert(1, "LMB push  RMB dig  hold A add  SPACE record goal")
        else:
            lines.insert(1, "LMB drag = DIG zone   RMB drag = DUMP zone   C clear")
        self.renderer.draw(e.ps, e.arm_state, contact_torques=e.contact_torques,
                           hud_lines=lines)

    # ── target construction ──────────────────────────────────────────
    def zones_target(self):
        hist = self.env.eval_hist_1d()
        grid = self.env.grid
        t = hist.astype(np.float64).copy()
        src = (grid >= min(self.dig)) & (grid <= max(self.dig))
        dst = (grid >= min(self.dump)) & (grid <= max(self.dump))
        if not src.any() or not dst.any():
            return None
        moved = t[src].sum() * 0.8
        t[src] *= 0.2
        t[dst] += moved / dst.sum()
        return (np.clip(t, 0, None) / t.sum()).astype(np.float32)

    # ── phases ───────────────────────────────────────────────────────
    def goal_reachability_note(self, target: np.ndarray) -> str:
        """Warn when the goal demands mass changes outside the workspace."""
        grid = self.env.grid_eval
        now = self.env.eval_hist_1d()
        outside = (grid < WORKSPACE[0]) | (grid > WORKSPACE[1])
        unreachable = float(np.abs(target - now)[outside].sum()) / 2.0
        if unreachable > 0.05:
            return (f"warning: {unreachable:.0%} of the change lies outside "
                    f"the arm's reach ({WORKSPACE[0]}-{WORKSPACE[1]} m)")
        return ""

    def _rehome_and_settle(self):
        """Skills are episodic behaviors from a settled bed + home arm pose;
        recreate those conditions between segments."""
        e = self.env.env
        e.arm = ArmDynamics(e.cfg)
        for _ in range(SEGMENT_SETTLE):
            e.solver.step(e.ps, e.cfg)
        e._refresh()
        self.obs = e._build_obs()

    def steer_gc(self, target: np.ndarray):
        """Direct goal-conditioned control: the policy takes the goal
        profile itself; episodic segments with arm re-homing."""
        note = self.goal_reachability_note(target)
        self._rehome_and_settle()
        best = w2_1d_exact(self.env.eval_hist_1d(), target, self.env.grid_eval)
        stall = 0
        for seg in range(MAX_SEGMENTS):
            for step in range(MPC_HORIZON * 2):
                for ev in self.pg.event.get():
                    if ev.type == self.pg.QUIT:
                        return "quit"
                    if ev.type == self.pg.KEYDOWN and ev.key == self.pg.K_ESCAPE:
                        return "aborted"
                self.obs, _, _, _, _ = self.env.step(
                    self.trainer.act(self.obs, target, deterministic=True))
                w2 = w2_1d_exact(self.env.eval_hist_1d(), target,
                                 self.env.grid_eval)
                self.status = (f"segment {seg + 1}/{MAX_SEGMENTS}  "
                               f"W2 {w2:.3f} m (best {best:.3f}) {note}")
                self.draw()
                if w2 < CONVERGED_W2:
                    return "converged"
            self._rehome_and_settle()
            w2 = w2_1d_exact(self.env.eval_hist_1d(), target, self.env.grid_eval)
            if w2 < best - 0.015:
                best, stall = w2, 0
            else:
                stall += 1
            if stall >= 3:
                return f"plateau at W2 {w2:.3f} m"
        return f"segment budget spent (W2 {best:.3f} m)"

    def steer(self, target: np.ndarray):
        """Skill-MPC: per segment, try every candidate skill in the sim's
        imagination, execute the one that actually reduces W2 the most."""
        if self.is_gc:
            return self.steer_gc(target)
        note = self.goal_reachability_note(target)
        self._rehome_and_settle()

        for seg in range(MAX_SEGMENTS):
            def show_planning(i, k, best_w2):
                self.status = (f"segment {seg + 1}/{MAX_SEGMENTS}: thinking "
                               f"{i}/{k} (best imagined W2 {best_w2:.3f}) {note}")
                self.draw()

            show_planning(0, len(self.mpc.candidates), float("nan"))
            z, predicted, baseline = self.mpc.plan(self.obs, target,
                                                   on_progress=show_planning)
            self.obs = self.env.env._build_obs()
            if z is None:
                return (f"no skill improves further (W2 {baseline:.3f} m — "
                        "skill-vocabulary limit)")
            for step in range(MPC_HORIZON):
                for ev in self.pg.event.get():
                    if ev.type == self.pg.QUIT:
                        return "quit"
                    if ev.type == self.pg.KEYDOWN and ev.key == self.pg.K_ESCAPE:
                        return "aborted"
                self.obs, _, _, _, _ = self.env.step(
                    self.trainer.act(self.obs, z, deterministic=True))
                w2 = w2_1d_exact(self.env.eval_hist_1d(), target,
                                 self.env.grid_eval)
                self.status = (f"segment {seg + 1}/{MAX_SEGMENTS}: executing  "
                               f"W2 {w2:.3f} m (predicted {predicted:.3f}) {note}")
                self.draw()
                if w2 < CONVERGED_W2:
                    return "converged"
            self._rehome_and_settle()
        w2 = w2_1d_exact(self.env.eval_hist_1d(), target, self.env.grid_eval)
        return f"segment budget spent (W2 {w2:.3f} m)"

    def run(self, max_frames=None):
        pg = self.pg
        drag_button = None
        drag_x0 = None
        frames = 0
        zero = np.zeros(self.env.act_dim, dtype=np.float32)
        while True:
            if max_frames is not None and frames >= max_frames:
                return
            frames += 1
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    return
                if ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_TAB:
                        self.mode = "ZONES" if self.mode == "SHAPE" else "SHAPE"
                        self.status = ("draw dig (LMB) and dump (RMB) zones"
                                       if self.mode == "ZONES"
                                       else "sculpt the soil, SPACE to record goal")
                    elif ev.key == pg.K_r:
                        self.reset_terrain()
                        self.status = "new terrain"
                    elif ev.key == pg.K_c and self.mode == "ZONES":
                        self.dig = self.dump = None
                        self.status = "zones cleared"
                    elif ev.key == pg.K_SPACE and self.mode == "SHAPE":
                        self.goal = self.env.eval_hist_1d().copy()
                        self.status = "GOAL recorded — now mess the soil up, ENTER to restore"
                    elif ev.key == pg.K_RETURN:
                        target = None
                        if self.mode == "SHAPE" and self.goal is not None:
                            target = self.goal
                        elif self.mode == "ZONES" and self.dig and self.dump:
                            target = self.zones_target()
                        if target is None:
                            self.status = ("record a goal first (SPACE)"
                                           if self.mode == "SHAPE"
                                           else "draw both zones first")
                        else:
                            outcome = self.steer(target)
                            if outcome == "quit":
                                return
                            self.status = f"done: {outcome} — keep editing or R for new terrain"
                if ev.type == pg.MOUSEBUTTONDOWN and ev.button in (1, 3):
                    drag_button = ev.button
                    drag_x0 = self.mouse_world()[0]
                if ev.type == pg.MOUSEBUTTONUP and ev.button == drag_button:
                    if self.mode == "ZONES" and drag_x0 is not None:
                        x1 = self.mouse_world()[0]
                        span = sorted((drag_x0, x1))
                        if span[1] - span[0] > 0.3:
                            if drag_button == 1:
                                self.dig = span
                            else:
                                self.dump = span
                    drag_button = None
                    drag_x0 = None

            wx, wy = self.mouse_world()
            keys = pg.key.get_pressed()
            if self.mode == "SHAPE":
                if drag_button == 1:
                    self.push_soil(wx, wy)
                elif drag_button == 3:
                    self.remove_soil(wx, wy)
                if keys[pg.K_a]:
                    self.add_soil(wx, wy)
            self.renderer.overlay_cursor = (
                (wx, wy, BRUSH_R, (200, 200, 255)) if self.mode == "SHAPE" else None)

            # Idle physics tick (arm holds position; soil settles/responds).
            self.obs, _, _, _, _ = self.env.step(zero)
            self.refresh_overlays()
            self.draw()


def main():
    args = parse_args()
    if args.selftest:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    app = App(args)
    if args.selftest:
        # Scripted pass: sculpt, record, perturb, steer briefly.
        app.push_soil(7.0, 3.0)
        for _ in range(30):
            app.obs, *_ = app.env.step(np.zeros(app.env.act_dim, np.float32))
        app.goal = app.env.eval_hist_1d().copy()
        app.remove_soil(6.0, 2.0)
        app.refresh_overlays()
        app.draw()
        app.mpc.candidates = app.mpc.candidates[:4]
        app.mpc.horizon = 40
        globals()["MPC_HORIZON"] = 40
        globals()["MAX_SEGMENTS"] = 2
        w0 = w2_1d_exact(app.env.eval_hist_1d(), app.goal, app.env.grid_eval)
        outcome = app.steer(app.goal)
        w1 = w2_1d_exact(app.env.eval_hist_1d(), app.goal, app.env.grid_eval)
        print(f"selftest ok: steer outcome = {outcome}; W2 {w0:.3f} -> {w1:.3f}")
        return
    app.run()


if __name__ == "__main__":
    main()
