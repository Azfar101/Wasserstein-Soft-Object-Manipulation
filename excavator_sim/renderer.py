"""
Pygame renderer for the soil + excavator scene.

Headless-safe and dual-mode:

* ``mode="human"``   — opens a window (interactive demo / live viewing).
* ``mode="rgb_array"`` — renders to an off-screen surface (uses the SDL
  ``dummy`` video driver if no display is configured) and returns an
  ``(H, W, 3) uint8`` frame, e.g. for video capture or vision-based policies.

The arm and bucket are drawn from the live arm state and the shared
:func:`excavator_sim.geometry.bucket_points`, so the rendered bucket and the
physics collider are guaranteed to match.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

from .config import SimConfig
from . import geometry
from . import hud as hud_mod


def _lerp(c1, c2, t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


class Renderer:
    def __init__(self, config: SimConfig, mode: str = "human"):
        import pygame
        self.pygame = pygame
        self.cfg = config
        self.mode = mode
        self.ppm = config.pixels_per_meter
        self.W = config.render.window_width
        self.H = config.render.window_height

        if mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("excavator_sim — 2D DEM soil")
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.Surface((self.W, self.H))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self._color_cache: Dict[int, tuple] = {}
        self._sprite_cache: Dict[tuple, object] = {}
        # Optional world-space polyline overlay [(x, y), ...] (e.g. a goal
        # terrain profile). Set by the caller; drawn each frame until cleared.
        self.overlay_polyline: Optional[List[tuple]] = None
        self.overlay_label: str = "target profile"
        # Optional shaded x-span zones [(x0, x1, (r, g, b), label), ...].
        self.overlay_zones: List[tuple] = []
        # Optional cursor disc (x, y, radius_m, (r, g, b)).
        self.overlay_cursor: Optional[tuple] = None

    # ── coordinate / colour helpers ─────────────────────────────────
    def _w2s(self, wx, wy):
        return int(wx * self.ppm), int((self.cfg.domain_height - wy) * self.ppm)

    def _particle_color(self, r):
        key = int(r * 1000)
        if key not in self._color_cache:
            pp = self.cfg.particle
            t = (r - pp.radius_min) / max(pp.radius_max - pp.radius_min, 1e-6)
            self._color_cache[key] = _lerp(self.cfg.render.color_small,
                                           self.cfg.render.color_large, t)
        return self._color_cache[key]

    def _sprite(self, pr, color):
        key = (pr, color)
        if key not in self._sprite_cache:
            pg = self.pygame
            size = max(pr * 2 + 1, 1)
            surf = pg.Surface((size, size), pg.SRCALPHA)
            pg.draw.circle(surf, color, (pr, pr), pr)
            self._sprite_cache[key] = surf
        return self._sprite_cache[key]

    def _boulder_sprite(self, pr):
        key = ("boulder", pr)
        if key not in self._sprite_cache:
            pg = self.pygame
            size = max(pr * 2 + 1, 1)
            surf = pg.Surface((size, size), pg.SRCALPHA)
            pg.draw.circle(surf, self.cfg.render.color_boulder, (pr, pr), pr)
            pg.draw.circle(surf, (50, 50, 60), (pr, pr), pr, 2)
            self._sprite_cache[key] = surf
        return self._sprite_cache[key]

    # ── scene elements ──────────────────────────────────────────────
    def _draw_grid(self):
        pg = self.pygame
        grid_c = (75, 75, 88)
        label_c = (110, 110, 130)
        for x_m in range(int(self.cfg.domain_width) + 1):
            sx, _ = self._w2s(x_m, 0.0)
            pg.draw.line(self.screen, grid_c, (sx, 0), (sx, self.H), 1)
            if x_m % 2 == 0:
                self.screen.blit(self.font.render(str(x_m), True, label_c),
                                 (sx + 2, self.H - 18))
        for y_m in range(int(self.cfg.domain_height) + 1):
            _, sy = self._w2s(0.0, y_m)
            pg.draw.line(self.screen, grid_c, (0, sy), (self.W, sy), 1)
            self.screen.blit(self.font.render(str(y_m), True, label_c), (2, sy + 2))

    def _draw_walls(self):
        pg = self.pygame
        c = self.cfg.render.wall_color
        pg.draw.line(self.screen, c, (0, self.H), (self.W, self.H), 3)
        pg.draw.line(self.screen, c, (0, 0), (0, self.H), 3)
        pg.draw.line(self.screen, c, (self.W - 1, 0), (self.W - 1, self.H), 3)

    def _draw_particles(self, ps):
        n = ps.count
        if n == 0:
            return
        sx = (ps.px[:n] * self.ppm).astype(np.int32)
        sy = ((self.cfg.domain_height - ps.py[:n]) * self.ppm).astype(np.int32)
        pr = np.clip((ps.radius[:n] * self.ppm).astype(np.int32), 1, 100)
        is_boul = ps.is_boulder[:n]
        blits = []
        for i in range(n):
            r = int(pr[i])
            if is_boul[i]:
                sprite = self._boulder_sprite(r)
            else:
                sprite = self._sprite(r, self._particle_color(ps.radius[i]))
            blits.append((sprite, (int(sx[i]) - r, int(sy[i]) - r)))
        self.screen.blits(blits, doreturn=False)

    def _beam(self, ax, ay, bx, by, half_w, fill, edge):
        """Draw a tapered beam (trapezoid) between two world points."""
        pg = self.pygame
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / ln, dx / ln
        poly = [
            self._w2s(ax + nx * half_w, ay + ny * half_w),
            self._w2s(ax - nx * half_w, ay - ny * half_w),
            self._w2s(bx - nx * half_w * 0.6, by - ny * half_w * 0.6),
            self._w2s(bx + nx * half_w * 0.6, by + ny * half_w * 0.6),
        ]
        pg.draw.polygon(self.screen, fill, poly)
        pg.draw.polygon(self.screen, edge, poly, 2)

    def _draw_arm(self, st: Dict[str, float], contact_torques: Optional[Dict]):
        pg = self.pygame
        YEL, DK_YEL = (215, 170, 20), (155, 115, 8)
        GRAY, LGRAY = (65, 65, 75), (145, 145, 155)
        a = self.cfg.arm
        pivot = (a.pivot_x, a.pivot_y)
        elbow = (st["elbow_x"], st["elbow_y"])
        wrist = (st["wrist_x"], st["wrist_y"])

        self._beam(*pivot, *elbow, 0.18, YEL, DK_YEL)
        self._beam(*elbow, *wrist, 0.11, YEL, DK_YEL)

        pts, k = geometry.bucket_points(
            wrist[0], wrist[1], st["mouth_ax"], st["mouth_ay"],
            st["stick_ax"], st["stick_ay"], self.cfg.bucket.radius)
        pg.draw.lines(self.screen, LGRAY, False, [self._w2s(*p) for p in pts], 3)
        pg.draw.line(self.screen, LGRAY, self._w2s(*k), self._w2s(*pts[0]), 3)

        # Teeth at the lip, fanned along the mouth direction.
        mouth_a = math.atan2(st["mouth_ay"], st["mouth_ax"])
        lip = pts[4]
        for frac in (-0.30, 0.0, 0.30):
            ta = mouth_a + frac * 0.6
            bx = lip[0] + 0.024 * math.cos(ta + math.pi * 0.5)
            by = lip[1] + 0.024 * math.sin(ta + math.pi * 0.5)
            pg.draw.line(self.screen, LGRAY, self._w2s(bx, by),
                         self._w2s(bx + 0.16 * math.cos(ta), by + 0.16 * math.sin(ta)), 2)

        for pt, col, rad in ((pivot, 8, 8), (elbow, 7, 7), (wrist, 6, 6)):
            s = self._w2s(*pt)
            pg.draw.circle(self.screen, GRAY, s, rad)
            pg.draw.circle(self.screen, LGRAY, s, rad, 2)

        if contact_torques is not None:
            gmax = self.cfg.render.torque_gauge_max
            self._torque_gauge(self._w2s(*pivot), contact_torques.get("shoulder", 0.0), gmax[0], "S")
            self._torque_gauge(self._w2s(*elbow), contact_torques.get("elbow", 0.0), gmax[1], "E")
            self._torque_gauge(self._w2s(*wrist), contact_torques.get("wrist", 0.0), gmax[2], "W")

    def _torque_gauge(self, pos, torque, max_torque, label):
        pg = self.pygame
        ratio = min(abs(torque) / max(max_torque, 1.0), 1.0)
        cx, cy = pos
        if abs(torque) < 1.0:
            pg.draw.circle(self.screen, (60, 60, 60), pos, 18, 1)
        else:
            if ratio < 0.5:
                t = ratio * 2.0
                color = (int(50 + 205 * t), int(220 - 20 * t), 50)
            else:
                t = (ratio - 0.5) * 2.0
                color = (255, int(200 * (1.0 - t)), int(50 * (1.0 - t)))
            sweep = ratio * math.pi * 1.5
            start = math.pi * 0.5 if torque > 0 else math.pi * 0.5 - sweep
            segs = max(6, int(sweep * 6))
            outer, inner = [], []
            for kk in range(segs + 1):
                ang = start + sweep * kk / segs
                outer.append((int(cx + 18 * math.cos(ang)), int(cy - 18 * math.sin(ang))))
                inner.append((int(cx + 12 * math.cos(ang)), int(cy - 12 * math.sin(ang))))
            poly = outer + inner[::-1]
            if len(poly) >= 3:
                pg.draw.polygon(self.screen, color, poly)
        lbl = self.font.render(label, True, self.cfg.render.ui_color)
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

    # ── main entry ──────────────────────────────────────────────────
    def draw(self, ps, arm_state, *, contact_torques=None, info=None,
             hud_lines: Optional[List[str]] = None, show_grid: bool = True):
        pg = self.pygame
        if self.mode == "human":
            pg.event.pump()
        self.screen.fill(self.cfg.render.bg_color)
        if show_grid:
            self._draw_grid()
        self._draw_walls()
        self._draw_particles(ps)
        if arm_state is not None:
            self._draw_arm(arm_state, contact_torques)

        for zx0, zx1, zc, zlabel in self.overlay_zones:
            sx0, _ = self._w2s(zx0, 0.0)
            sx1, _ = self._w2s(zx1, 0.0)
            band = pg.Surface((max(sx1 - sx0, 1), self.H), pg.SRCALPHA)
            band.fill((*zc, 46))
            self.screen.blit(band, (sx0, 0))
            self.screen.blit(self.font.render(zlabel, True, zc),
                             (sx0 + 4, self.H - 40))

        if self.overlay_polyline:
            pts = [self._w2s(x, y) for x, y in self.overlay_polyline]
            if len(pts) >= 2:
                pg.draw.lines(self.screen, (255, 70, 70), False, pts, 3)
                lbl = self.font.render(self.overlay_label, True, (255, 90, 90))
                self.screen.blit(lbl, (pts[0][0] + 4, max(pts[0][1] - 20, 2)))

        if self.overlay_cursor is not None:
            cx, cy, cr, cc = self.overlay_cursor
            sx, sy = self._w2s(cx, cy)
            pg.draw.circle(self.screen, cc, (sx, sy),
                           max(int(cr * self.ppm), 2), 2)

        fps = self.clock.get_fps()
        if hud_lines is None and info is not None:
            hud_lines = hud_mod.compact_overlay(info, fps)
        if hud_lines:
            y = 8
            for line in hud_lines:
                self.screen.blit(self.font.render(line, True, self.cfg.render.ui_color), (8, y))
                y += 18

        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(self.cfg.render.fps_cap)
            return None
        self.clock.tick(self.cfg.render.fps_cap)
        return np.transpose(pg.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        self.pygame.quit()
