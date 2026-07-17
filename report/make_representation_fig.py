"""
Visualize the full state-representation pipeline on one real sim state:

  (a) the world       — rendered simulator frame (~1200 particles + arm)
  (b) policy obs      — the 35 numbers pi(a|s,z) actually receives
  (c) external measure— the 64-bin soil mass histogram mu (all phi sees)
  (d) latent phi(mu)  — where this state sits on the learned transport map,
                        with real skill-episode trajectories for context

Real data end to end: live env, trained checkpoint (tmsd_w2_rt_d4).

    python report/make_representation_fig.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

C1 = "#2a78d6"; C2 = "#1baf7a"; C3 = "#eda100"; C6 = "#e34948"
MUTED = "#9a9992"; INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": "#d8d7d0", "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "pdf.fonttype": 42,
})

SEED = 123
ckpt = ROOT / "runs" / "tmsd_w2_rt_d4" / "ckpt_latest.pt"
saved = torch.load(ckpt, map_location="cpu", weights_only=False)
cfg: TMSDConfig = saved["cfg"]

env = SkillDiscoveryEnv(hist_bins=cfg.hist_bins, max_episode_steps=10**9,
                        randomize_terrain=True, render_mode="rgb_array")
tr = TMSDTrainer(cfg, env.grid)
tr.load(str(ckpt))

obs, hist, _ = env.reset(seed=SEED)
frame = env.render()

# phi trajectories for 4 skill directions (for panel d context)
dirs = [np.array(v, np.float32) for v in
        ([1, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0])]
labels = ["z = +e1", "z = -e1", "z = +e3", "z = -e3"]
colors = [C1, C6, C2, C3]
trajs = []
for z in dirs:
    o, h, _ = env.reset(seed=SEED)
    hs = [h.copy()]
    for _ in range(200):
        o, h, term, trunc, _ = env.step(tr.act(o, z, deterministic=True))
        hs.append(h.copy())
        if term or trunc:
            break
    with torch.no_grad():
        phis = tr.phi(torch.as_tensor(np.stack(hs), dtype=torch.float32,
                                      device=tr.device)).cpu().numpy()
    trajs.append(phis)
env.close()

with torch.no_grad():
    phi0 = tr.phi(torch.as_tensor(hist, dtype=torch.float32,
                                  device=tr.device).unsqueeze(0)).cpu().numpy()[0]

# ── figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.25)

# (a) world
ax = fig.add_subplot(gs[0, 0])
ax.imshow(frame)
ax.set_title("(a)  The world: ~1200 simulated grains + 3-joint arm")
ax.axis("off")

# (b) policy observation: 35 numbers
ax = fig.add_subplot(gs[0, 1])
groups = [("joints\n(pos+vel)", 6, C2, 1.32), ("wrist\n+mouth", 3, C3, 1.05),
          ("loads, payload,\ncontact, forces", 7, C6, 1.32),
          ("16-bin soil heightfield", 16, C1, 1.10), ("boulder\n(off)", 3, MUTED, 1.32)]
x0 = 0
for name, n, c, ty in groups:
    vals = obs[x0:x0 + n]
    ax.bar(np.arange(x0, x0 + n), vals, width=0.85, color=c)
    ax.text(x0 + n / 2 - 0.5, ty, name, ha="center", fontsize=6.8, color=c)
    x0 += n
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlim(-1, 35)
ax.set_ylim(-1.15, 1.62)
ax.set_xlabel("observation index (35 total)")
ax.set_title("(b)  What the POLICY sees: body + coarse soil (35-dim)")

# (c) external measure
ax = fig.add_subplot(gs[1, 0])
ax.bar(env.grid, hist, width=env.grid[1] - env.grid[0], color=C1, alpha=0.85)
ax.set_xlabel("x (m)")
ax.set_ylabel("soil mass fraction")
ax.set_title("(c)  What the OBJECTIVE sees: $\\mu$ = 64-bin soil mass\n"
             "distribution -- zero robot information, sums to 1")
ax.text(0.98, 0.92, f"$\\sum_i \\mu_i$ = {hist.sum():.3f}",
        transform=ax.transAxes, ha="right", fontsize=8, color=INK2)

# (d) latent map
ax = fig.add_subplot(gs[1, 1])
for phis, lbl, c in zip(trajs, labels, colors):
    ax.plot(phis[:, 0], phis[:, 1], color=c, lw=1.8, alpha=0.9)
    ax.scatter(phis[-1, 0], phis[-1, 1], color=c, marker="*", s=110,
               zorder=4, edgecolors="white", linewidths=0.5)
    txt = ax.annotate(lbl, (phis[-1, 0], phis[-1, 1]), fontsize=7.5,
                      color=c, xytext=(4, 5), textcoords="offset points")
    txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground=SURF)])
ax.scatter(*phi0[:2], color=INK, s=45, zorder=5)
ax.annotate("this state:\n$\\phi(\\mu)$", phi0[:2], fontsize=7.5, color=INK,
            xytext=(6, -16), textcoords="offset points")
ax.set_xlabel("$\\phi_1$")
ax.set_ylabel("$\\phi_2$")
ax.set_title("(d)  The learned map: $\\phi(\\mu) \\in \\mathbb{R}^4$ (first 2 dims)\n"
             "skills travel directions; distance $\\approx$ physical transport")

fig.suptitle("One state, four representations: from physics to the transport map",
             fontsize=12.5, fontweight="bold", y=0.995)
out = ROOT / "report" / "figures"
fig.savefig(out / "fig_representation.pdf", bbox_inches="tight")
fig.savefig(out / "fig_representation.png", dpi=150, bbox_inches="tight")
print(f"saved {out / 'fig_representation.pdf'} (+ png)")
