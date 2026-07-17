"""
Pairwise-W2 coverage matrices for one representative checkpoint per method
family, same 8 protocol skills, same terrain, SHARED color scale (auto-scaled
heatmaps would hide the cross-method differences).

Output: report/figures/fig_coverage_matrices.{pdf,png}

    python report/make_coverage_matrices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmsd.metrics import w2_1d_exact
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

SEED = 123
N_SKILLS = 8
STEPS = 200

METHODS = [
    ("tmsd_w2_rt_d4", "OURS\ndist-max + soil $\\phi$"),
    ("abl_fullstate_temporal", "METRA (vanilla)\ndist-max + full $\\phi$"),
    ("mi_soil_s0", "DIAYN-style\nMI + soil $\\phi$"),
    ("dads_soil_s0", "DADS-style\ndynamics + soil $\\phi$"),
]

INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "font.size": 8.5, "axes.titlesize": 9,
    "pdf.fonttype": 42,
})


def coverage_matrix(run_name):
    ckpt = ROOT / "runs" / run_name / "ckpt_latest.pt"
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    env = SkillDiscoveryEnv(hist_bins=cfg.hist_bins, max_episode_steps=STEPS,
                            randomize_terrain=True)
    tr = TMSDTrainer(cfg, env.grid)
    tr.load(str(ckpt))
    rng = np.random.default_rng(SEED)
    v = rng.normal(size=(N_SKILLS, cfg.skill_dim))
    zs = (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)

    terms = []
    for z in zs:
        obs, hist, _ = env.reset(seed=SEED)
        for _ in range(STEPS):
            obs, hist, term, trunc, _ = env.step(tr.act(obs, z, deterministic=True))
            if term or trunc:
                break
        terms.append(env.eval_hist_1d())
    grid = env.grid_eval
    env.close()

    M = np.zeros((N_SKILLS, N_SKILLS))
    for i in range(N_SKILLS):
        for j in range(i + 1, N_SKILLS):
            M[i, j] = M[j, i] = w2_1d_exact(terms[i], terms[j], grid)
    return M


def main():
    mats = {}
    for run, label in METHODS:
        print(f"rolling out {run} ...")
        mats[label] = coverage_matrix(run)
    vmax = max(M.max() for M in mats.values())

    fig, axes = plt.subplots(1, len(METHODS), figsize=(11.6, 3.4))
    for ax, (label, M) in zip(axes, mats.items()):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
        off = M[~np.eye(N_SKILLS, dtype=bool)]
        ax.set_title(f"{label}\nmean pairwise = {off.mean():.3f} m",
                     fontsize=8.5)
        ax.set_xticks(range(N_SKILLS))
        ax.set_yticks(range(N_SKILLS))
        ax.set_xlabel("skill")
        if ax is axes[0]:
            ax.set_ylabel("skill")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("$W_2$ between terminal soil states (m)")
    fig.suptitle("Coverage matrices, one shared color scale: same 8 skills, "
                 "same terrain, every method", fontsize=11.5, fontweight="bold")
    out = ROOT / "report" / "figures"
    fig.savefig(out / "fig_coverage_matrices.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_coverage_matrices.png", dpi=150, bbox_inches="tight")
    print(f"saved {out / 'fig_coverage_matrices.pdf'} (+png)")


if __name__ == "__main__":
    main()
