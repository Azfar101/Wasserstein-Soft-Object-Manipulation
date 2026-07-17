"""
Public showcase: three-act auto-running demo of TMSD skills.

  Act 1  "Nobody taught it to dig"   — skill vocabulary, captioned
  Act 2  "Without our ingredient"    — side-by-side vs arm-waving baseline
  Act 3  "Give it a job"             — zone excavation with goal overlay

Runs unattended (~2 min). ESC or window close exits any time.

Usage:
    python scripts/showcase.py
    python scripts/showcase.py --selftest      # headless, seconds
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmsd.metrics import w2_1d_exact
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 720
CAP_H = 64
SEED = 123

ACT1_SKILLS = [
    (np.array([+0.89, -0.38, -0.18, +0.20], np.float32), "Skill A: dig and pile RIGHT"),
    (np.array([-0.73, +0.27, +0.51, +0.37], np.float32), "Skill B: carve and push LEFT"),
    (np.array([+0.41, -0.08, +0.70, +0.58], np.float32), "Skill C: excavate the middle"),
    (np.array([+0.29, +0.00, -0.27, -0.92], np.float32), "Skill D: wide shallow cut"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, default="tmsd_w2_rt_d4")
    p.add_argument("--baseline", type=str, default="abl_fullstate_temporal")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def load(run_name):
    path = ROOT / "runs" / run_name / "ckpt_latest.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    env = SkillDiscoveryEnv(hist_bins=cfg.hist_bins,
                            max_episode_steps=10 ** 9,
                            randomize_terrain=True, render_mode="rgb_array")
    tr = TMSDTrainer(cfg, env.grid)
    tr.load(str(path))
    return env, tr


class Stage:
    """Owns the window; blits env frames with a caption bar."""

    def __init__(self, headless: bool):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((W, H + CAP_H))
        pygame.display.set_caption("TMSD — skills nobody taught")
        self.font_big = pygame.font.SysFont("segoeui", 30, bold=True)
        self.font = pygame.font.SysFont("segoeui", 20)
        self.clock = pygame.time.Clock()

    def alive(self) -> bool:
        for ev in self.pg.event.get():
            if ev.type == self.pg.QUIT:
                return False
            if ev.type == self.pg.KEYDOWN and ev.key == self.pg.K_ESCAPE:
                return False
        return True

    def _surf(self, frame: np.ndarray, size):
        s = self.pg.surfarray.make_surface(frame.swapaxes(0, 1))
        return self.pg.transform.smoothscale(s, size)

    def show(self, title, subtitle, frame=None, frames2=None, fps=60):
        self.screen.fill((16, 16, 20))
        if frame is not None:
            self.screen.blit(self._surf(frame, (W, H)), (0, CAP_H))
        elif frames2 is not None:
            half = W // 2 - 4
            hh = int(half * H / W)
            y = CAP_H + (H - hh) // 2
            (fa, label_a), (fb, label_b) = frames2
            self.screen.blit(self._surf(fa, (half, hh)), (0, y))
            self.screen.blit(self._surf(fb, (half, hh)), (half + 8, y))
            for x, text, col in ((10, label_a, (120, 230, 140)),
                                 (half + 18, label_b, (240, 120, 110))):
                self.screen.blit(self.font.render(text, True, col), (x, y + 8))
        self.screen.blit(self.font_big.render(title, True, (235, 235, 240)), (16, 4))
        self.screen.blit(self.font.render(subtitle, True, (170, 170, 185)), (16, 38))
        self.pg.display.flip()
        self.clock.tick(fps)

    def hold(self, title, subtitle, frame, seconds):
        for _ in range(int(seconds * 30)):
            if not self.alive():
                return False
            self.show(title, subtitle, frame=frame, fps=30)
        return True


def act1(stage, env, tr, steps):
    title = "Act 1 — Nobody taught it to dig"
    for z, label in ACT1_SKILLS:
        obs, _, _ = env.reset(seed=SEED)
        zz = z[:tr.cfg.skill_dim] / np.linalg.norm(z[:tr.cfg.skill_dim])
        for _ in range(steps):
            if not stage.alive():
                return False
            obs, _, _, _, _ = env.step(tr.act(obs, zz, deterministic=True))
            stage.show(title, f"{label}   (trained with zero rewards, "
                              f"zero demonstrations)", frame=env.render())
    return True


def act2(stage, env_a, tr_a, env_b, tr_b, steps):
    title = "Act 2 — Same training, without our key ingredient"
    sub = ("LEFT: representation sees only the SOIL -> it digs.   "
           "RIGHT: sees its own body -> it just... dances.")
    obs_a, _, _ = env_a.reset(seed=SEED)
    obs_b, _, _ = env_b.reset(seed=SEED)
    za = ACT1_SKILLS[0][0][:tr_a.cfg.skill_dim]
    za = za / np.linalg.norm(za)
    zb = ACT1_SKILLS[0][0][:tr_b.cfg.skill_dim]
    zb = zb / np.linalg.norm(zb)
    for _ in range(steps):
        if not stage.alive():
            return False
        obs_a, *_ = env_a.step(tr_a.act(obs_a, za, deterministic=True))
        obs_b, *_ = env_b.step(tr_b.act(obs_b, zb, deterministic=True))
        stage.show(title, sub, frames2=((env_a.render(), "ours: soil-only view"),
                                        (env_b.render(), "baseline: full view")))
    return True


def act3(stage, env, tr, steps):
    title = "Act 3 — Now give it a job"
    dig, dump = (7.0, 9.0), (4.5, 6.5)
    obs, _, _ = env.reset(seed=SEED)
    before = env.render()
    hist = env.eval_hist_1d()
    grid = env.grid_eval
    t = hist.astype(np.float64).copy()
    src = (grid >= dig[0]) & (grid <= dig[1])
    dst = (grid >= dump[0]) & (grid <= dump[1])
    moved = t[src].sum() * 0.8
    t[src] *= 0.2
    t[dst] += moved / dst.sum()
    target = (np.clip(t, 0, None) / t.sum()).astype(np.float32)

    r = env.env._renderer
    heights = env.hist_to_heights(target)
    r.overlay_polyline = list(zip(env.grid, heights))
    r.overlay_label = "ORDER: dig right zone, dump left (red line = goal)"

    tgt_t = torch.as_tensor(target, device=tr.device).unsqueeze(0)
    dig0 = float(hist[src].sum())
    z = None
    for step in range(steps):
        if not stage.alive():
            return False
        if step % 10 == 0:
            with torch.no_grad():
                h = torch.as_tensor(env._hist(), device=tr.device).unsqueeze(0)
                dz = (tr.phi(tgt_t) - tr.phi(h)).squeeze(0).cpu().numpy()
            n = np.linalg.norm(dz)
            z = (dz / n if n > 1e-8 else tr.sample_skill()).astype(np.float32)
        obs, *_ = env.step(tr.act(obs, z, deterministic=True))
        cur = env.eval_hist_1d()
        cleared = 1.0 - float(cur[src].sum()) / max(dig0, 1e-9)
        stage.show(title, f"excavating the ordered zone...  "
                          f"{max(cleared, 0):.0%} of the zone's soil removed",
                   frame=env.render())
    after = env.render()
    cur = env.eval_hist_1d()
    cleared = 1.0 - float(cur[src].sum()) / max(dig0, 1e-9)
    ok = stage.hold("Before", "the terrain as it started", before, 2.0)
    ok = ok and stage.hold(
        "After", f"zone cleared: {cleared:.0%} — commanded by a goal state, "
                 "executed by self-taught skills", after, 4.0)
    return ok


def main():
    args = parse_args()
    stage = Stage(headless=args.selftest)
    env_a, tr_a = load(args.run_name)
    env_b, tr_b = load(args.baseline)

    s1, s2, s3 = (200, 260, 600) if not args.selftest else (5, 5, 8)
    ok = act1(stage, env_a, tr_a, s1)
    ok = ok and act2(stage, env_a, tr_a, env_b, tr_b, s2)
    ok = ok and act3(stage, env_a, tr_a, s3)
    if ok and not args.selftest:
        stage.hold("TMSD — Transport-Metric Skill Discovery",
                   "all behaviors emerged without a single reward. "
                   "ESC to close.", env_a.render(), 6.0)
    env_a.close()
    env_b.close()
    if args.selftest:
        print("showcase selftest ok")


if __name__ == "__main__":
    main()
