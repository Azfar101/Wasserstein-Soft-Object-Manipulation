"""
Generate all report figures as vector PDFs from real training data.

Output: report/figures/*.pdf  (+ copies of existing rollout PNGs)
Rerunnable; reads runs/*.csv only (no simulation, no GPU).

    python report/make_figures.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (fixed slot order — never cycled).
C1_BLUE = "#2a78d6"
C2_AQUA = "#1baf7a"
C3_YELLOW = "#eda100"
C5_VIOLET = "#4a3aa7"
C6_RED = "#e34948"
MUTED = "#9a9992"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d0", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e8e7e0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "pdf.fonttype": 42,
})


def read_eps(run, key="terminal_w2_from_start"):
    p = RUNS / run / "episodes.csv"
    return np.array([float(r[key]) for r in csv.DictReader(open(p))])


def read_log(run, key):
    p = RUNS / run / "log.csv"
    out = [(int(r["step"]), float(r[key])) for r in csv.DictReader(open(p))
           if r.get(key) not in (None, "")]
    return np.array(out).T


def rolling(x, k):
    return np.convolve(x, np.ones(k) / k, mode="valid")


# ── 1. learning curve: skills emerge from zero rewards ───────────────
def fig_learning_curve():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    w2 = read_eps("tmsd_w2_s0")
    ax.plot(np.arange(len(w2))[:len(rolling(w2, 100))] + 50, rolling(w2, 100),
            color=C1_BLUE, lw=2)
    ax.fill_between(np.arange(len(w2))[:len(rolling(w2, 100))] + 50,
                    0, rolling(w2, 100), color=C1_BLUE, alpha=0.10, lw=0)
    ax.annotate("digging emerges", xy=(700, 0.16), xytext=(1050, 0.09),
                color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("training episode")
    ax.set_ylabel("soil moved per episode ($W_2$, m)")
    ax.set_title("Reward-free training: soil transport per episode")
    ax.set_xlim(50, len(w2))
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curve.pdf")


# ── 2. frontier coverage bars (the headline result) ─────────────────
def fig_frontier_coverage():
    # (label, per-seed coverages, works?)
    data = [
        ("$W_2$\n+ soil", [0.359, 0.288, 0.330], True),
        ("Euclid\n+ soil", [0.376, 0.313, 0.368], True),
        ("temporal\n+ soil", [0.391, 0.203, 0.278], True),
        ("METRA\n(vanilla)", [0.090, 0.028, 0.001], False),
        ("DIAYN\n+ soil", [0.034, 0.001, 0.026], False),
        ("DIAYN\n(vanilla)", [0.020, 0.001, 0.070], False),
        ("DADS\n+ soil", [0.000, 0.015], False),
        ("DADS\n(vanilla)", [0.057, 0.203], False),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    xs = np.arange(len(data))
    for x, (label, vals, works) in zip(xs, data):
        m, sd = np.mean(vals), np.std(vals)
        ax.bar(x, m, width=0.62, color=C1_BLUE if works else MUTED,
               yerr=sd, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        ax.scatter([x] * len(vals), vals, s=8, color=INK2, zorder=3, alpha=0.7)
        ax.text(x + 0.36, m + 0.005, f"{m:.2f}", ha="left", fontsize=7.5,
                color=INK)
    ax.axhline(0.083, color=C6_RED, ls="--", lw=1)
    ax.text(len(data) - 0.4, 0.093, "ambient settling floor", ha="right",
            fontsize=7.5, color=C6_RED)
    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in data], fontsize=7.5)
    ax.set_ylabel("skill coverage (pairwise $W_2$, m)")
    ax.set_title("Frontier comparison -- identical simulation, budget, frozen evaluation")
    ax.set_ylim(0, 0.47)
    # Bracket over the working family: the recipe is the claim, the three
    # metric variants inside it are statistically tied (that tie is a finding).
    ax.plot([-0.35, 2.35], [0.435, 0.435], color=C1_BLUE, lw=1.2)
    ax.plot([-0.35, -0.35], [0.425, 0.435], color=C1_BLUE, lw=1.2)
    ax.plot([2.35, 2.35], [0.425, 0.435], color=C1_BLUE, lw=1.2)
    ax.text(1.0, 0.445, "OUR RECIPE: distance-max objective + soil-only $\\phi$"
            "\n(ground metric interchangeable: 3-way tie)",
            ha="center", va="bottom", fontsize=7.5, color=C1_BLUE)
    ax.text(5.5, 0.435, "prior methods & wrong conditioning",
            ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_ylim(0, 0.52)
    fig.tight_layout()
    fig.savefig(OUT / "fig_frontier_coverage.pdf")


# ── 3. goal-reaching contraction curves ───────────────────────────────
def fig_contraction():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    rows = list(csv.DictReader(open(RUNS / "tmsd_w2_rt_d4" / "demo_shaping.csv")))
    colors = {"trench right, spoil left": C1_BLUE, "central berm": C2_AQUA,
              "flatten workspace": C3_YELLOW}
    for tgt, col in colors.items():
        pts = [(int(r["step"]), float(r["w2_to_goal"])) for r in rows
               if r["target"] == tgt]
        s, w = zip(*pts)
        ax.plot(s, w, color=col, lw=2, marker="o", ms=2.5)
        ax.text(s[-1] + 6, w[-1], tgt, fontsize=7.5, color=col, va="center")
    ax.axhspan(0, 0.06, color=MUTED, alpha=0.18, lw=0)
    ax.text(6, 0.03, "ambient noise", fontsize=7, color=INK2, va="center")
    ax.annotate("skill-granularity plateau $r(\\varepsilon)$",
                xy=(300, 0.23), xytext=(160, 0.36), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("environment steps")
    ax.set_ylabel("$W_2$ to commanded profile (m)")
    ax.set_title("Zero-shot goal-reaching: monotone contraction to $r(\\varepsilon)$")
    ax.set_xlim(0, 520)
    ax.set_ylim(0, 0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_contraction.pdf")


# ── 4. mechanism small-multiples: telescope vs saturate vs starve ────
def fig_mechanisms():
    panels = [
        ("abl_temporal_rt_d4", "distance-max + soil  (telescopes: must keep moving soil)",
         C1_BLUE, "sac/reward_mean"),
        ("mi_soil_s0", "MI (DIAYN-style) + soil  (saturates: identifiable without change)",
         C5_VIOLET, "sac/reward_mean"),
        ("dads_soil_s0", "dynamics (DADS-style) + soil  (starves: slow state $\\Rightarrow$ reward $\\equiv$ 0)",
         C6_RED, "sac/reward_mean"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 4.4), sharex=True)
    for ax, (run, title, col, key) in zip(axes, panels):
        steps, r = read_log(run, key)
        ax.plot(steps / 1000, r, color=col, lw=1.6)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylabel("reward/step", fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(min(0, r.min() * 1.1), max(r.max() * 1.15, 0.01))
    axes[-1].set_xlabel("training steps (thousands)")
    fig.suptitle("Three objective families on the same slow material state",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig_mechanisms.pdf")


# ── 5. reward per meter of soil (log, 'lie detector') ────────────────
def fig_reward_per_meter():
    data = [
        ("dist-max Euclid + soil", 36, True),
        ("dist-max $W_2$ + soil", 170, True),
        ("DADS (full state)", 1100, False),
        ("dist-max temporal + soil", 12800, True),
        ("METRA (full state)", 300000, False),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ys = np.arange(len(data))
    for y, (label, v, works) in zip(ys, data):
        ax.barh(y, v, height=0.6, color=C1_BLUE if works else MUTED)
        ax.text(v * 1.25, y, f"{v:,.0f}", va="center", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=8)
    ax.set_xlabel("episode reward per meter of soil actually moved (log)")
    ax.set_title("Reward earned per meter of soil moved (high = paid for dancing)")
    ax.set_xlim(10, 2e6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reward_per_meter.pdf")


# ── 6. factorization effect across seeds ──────────────────────────────
def fig_factorization():
    soil = [0.359, 0.288, 0.330, 0.376, 0.313, 0.368, 0.391, 0.203, 0.278]
    full = [0.090, 0.028, 0.001]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for x, vals, col, label in ((0, soil, C1_BLUE, "$\\phi$(soil only)\n9 runs"),
                                (1, full, MUTED, "$\\phi$(full state)\n3 runs")):
        ax.bar(x, np.mean(vals), width=0.55, color=col,
               yerr=np.std(vals), error_kw=dict(ecolor=INK2, lw=1, capsize=3))
        ax.scatter(np.full(len(vals), x) + np.linspace(-0.1, 0.1, len(vals)),
                   vals, s=10, color=INK2, alpha=0.75, zorder=3)
        ax.text(x, np.mean(vals) + np.std(vals) + 0.02,
                f"{np.mean(vals):.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$\\phi$(soil only)", "$\\phi$(full state)"], fontsize=9)
    ax.set_ylabel("skill coverage (m)")
    ax.set_title("The 8$\\times$ factorization effect")
    fig.tight_layout()
    fig.savefig(OUT / "fig_factorization.pdf")


# ── 7. copy existing rollout images ───────────────────────────────────
def copy_rollout_pngs():
    for src, dst in [
        (RUNS / "tmsd_w2_rt_d4" / "skills_heightfields.png", "fig_terminal_profiles.png"),
        (RUNS / "tmsd_w2_rt_d4" / "skills_coverage.png", "fig_coverage_matrix.png"),
        (RUNS / "tmsd_w2_rt_d4" / "skills_latent.png", "fig_phi_trajectories.png"),
        (RUNS / "tmsd_w2_rt_d4" / "demo_shaping.png", "fig_demo_shaping.png"),
        (RUNS / "tmsd_w2_s0" / "skills_heightfields.png", "fig_terminal_profiles_pilot.png"),
    ]:
        if src.exists():
            shutil.copy(src, OUT / dst)
            print(f"copied {dst}")


if __name__ == "__main__":
    fig_learning_curve()
    fig_frontier_coverage()
    fig_contraction()
    fig_mechanisms()
    fig_reward_per_meter()
    fig_factorization()
    copy_rollout_pngs()
    print(f"figures -> {OUT}")
