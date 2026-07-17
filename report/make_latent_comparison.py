"""
Per-method comparison: skill LATENT trajectories (each model's own phi)
vs. physical skill COVERAGE (terminal soil profiles), same 8 protocol
skills, same terrain, for one representative checkpoint per family.

Output: report/figures/fig_latent_coverage_comparison.{pdf,png}

    python report/make_latent_comparison.py
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
    ("tmsd_w2_rt_d4", "OURS: distance-max + soil-only $\\phi$", "#2a78d6"),
    ("abl_fullstate_temporal", "METRA (vanilla): distance-max + full-state $\\phi$", "#9a9992"),
    ("mi_soil_s0", "DIAYN-style: MI + soil-only $\\phi$", "#4a3aa7"),
    ("dads_soil_s0", "DADS-style: dynamics + soil-only $\\phi$", "#e34948"),
]
SKILL_COLORS = plt.get_cmap("hsv")

INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": "#d8d7d0",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8.5, "axes.titlesize": 9, "pdf.fonttype": 42,
})


def rollout_method(run_name):
    ckpt = ROOT / "runs" / run_name / "ckpt_latest.pt"
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    phi_on_obs = getattr(cfg, "phi_input", "hist") == "obs"
    env = SkillDiscoveryEnv(hist_bins=cfg.hist_bins, max_episode_steps=STEPS,
                            randomize_terrain=True)
    tr = TMSDTrainer(cfg, env.grid)
    tr.load(str(ckpt))

    rng = np.random.default_rng(SEED)
    v = rng.normal(size=(N_SKILLS, cfg.skill_dim))
    zs = (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)

    latents, terminals, initial = [], [], None
    for z in zs:
        obs, hist, _ = env.reset(seed=SEED)
        if initial is None:
            initial = env.eval_hist_1d()
        traj = [obs.copy() if phi_on_obs else hist.copy()]
        for _ in range(STEPS):
            obs, hist, term, trunc, _ = env.step(tr.act(obs, z, deterministic=True))
            traj.append(obs.copy() if phi_on_obs else hist.copy())
            if term or trunc:
                break
        with torch.no_grad():
            phis = tr.phi(torch.as_tensor(np.stack(traj), dtype=torch.float32,
                                          device=tr.device)).cpu().numpy()
        latents.append(phis)
        terminals.append(env.eval_hist_1d())
    grid = env.grid_eval
    env.close()

    n = len(terminals)
    cov = np.mean([w2_1d_exact(terminals[i], terminals[j], grid)
                   for i in range(n) for j in range(i + 1, n)])
    return latents, terminals, initial, grid, cov


def main():
    fig, axes = plt.subplots(len(METHODS), 2, figsize=(9.6, 3.0 * len(METHODS)))
    for row, (run, label, accent) in enumerate(METHODS):
        print(f"rolling out {run} ...")
        latents, terminals, initial, grid, cov = rollout_method(run)
        ax_l, ax_c = axes[row]

        # latent panel: center each trajectory at its own start for comparability
        for k, P in enumerate(latents):
            c = SKILL_COLORS(k / N_SKILLS)
            Q = P - P[0]
            ax_l.plot(Q[:, 0], Q[:, 1], color=c, lw=1.4, alpha=0.9)
            ax_l.scatter(Q[-1, 0], Q[-1, 1], color=c, marker="*", s=70,
                         zorder=4, edgecolors="white", linewidths=0.4)
        ax_l.scatter(0, 0, color=INK, s=25, zorder=5)
        ax_l.set_title(f"{label}\nskill latent $\\phi$ (start-centered)",
                       color=accent, fontweight="bold")
        ax_l.set_xlabel("$\\phi_1 - \\phi_1(0)$")
        ax_l.set_ylabel("$\\phi_2 - \\phi_2(0)$")
        ax_l.axhline(0, color="#e8e7e0", lw=0.6, zorder=0)
        ax_l.axvline(0, color="#e8e7e0", lw=0.6, zorder=0)

        # coverage panel: terminal soil profiles
        ax_c.plot(grid, initial, "k--", lw=1.4, label="initial terrain")
        for k, tprof in enumerate(terminals):
            ax_c.plot(grid, tprof, color=SKILL_COLORS(k / N_SKILLS), lw=1.1)
        ax_c.set_title(f"terminal soil per skill --- coverage = {cov:.3f} m",
                       fontweight="bold")
        ax_c.set_xlabel("x (m)")
        ax_c.set_ylabel("soil mass fraction")
        if row == 0:
            ax_c.legend(fontsize=7, loc="upper right")

    fig.suptitle("Same 8 skills, same terrain: each method's latent map vs.\n"
                 "what its skills physically do to the soil",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = ROOT / "report" / "figures"
    fig.savefig(out / "fig_latent_coverage_comparison.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_latent_coverage_comparison.png", dpi=140,
                bbox_inches="tight")
    print(f"saved {out / 'fig_latent_coverage_comparison.pdf'} (+png)")


if __name__ == "__main__":
    main()
