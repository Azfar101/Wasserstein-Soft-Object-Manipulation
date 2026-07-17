"""
Demo figure: zero-shot soil shaping to user-specified target profiles.

Takes analytic, mass-conserving target profiles (trench / berm / two-hill),
steers with the METRA-style zero-shot rule z = (phi(goal) - phi(now)) /
||.|| using a trained soil-phi checkpoint, and renders a 3-panel figure:

  A. initial vs target vs achieved soil profile, per target
  B. W2-to-goal over control steps (contraction curves)
  C. rendered sim frames at start / middle / end, per target

Output: runs/<run-name>/demo_shaping.png (+ demo_shaping.csv)

Usage:
    python scripts/demo_shaping.py --run-name tmsd_w2_rt_d4 --steps 400
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmsd.metrics import w2_1d_exact
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, default="tmsd_w2_rt_d4")
    p.add_argument("--ckpt", type=str, default="ckpt_latest.pt")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--replan-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--live", action="store_true",
                   help="open the sim window and steer live (no figure)")
    return p.parse_args()


def _renorm(h: np.ndarray) -> np.ndarray:
    h = np.clip(h, 0.0, None)
    return (h / h.sum()).astype(np.float32)


def make_targets(init: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    """Analytic mass-conserving edits of the initial profile, confined to
    the arm's effective workspace (x ~ 5-9.5 m)."""
    x = grid
    targets = {}

    # 1. Trench at x in [7.2, 8.8], spoil piled to the left [5.2, 6.8].
    t = init.copy().astype(np.float64)
    cut = (x > 7.2) & (x < 8.8)
    dst = (x > 5.2) & (x < 6.8)
    moved = t[cut].sum() * 0.85
    t[cut] *= 0.15
    t[dst] += moved / dst.sum()
    targets["trench right, spoil left"] = _renorm(t)

    # 2. Berm: concentrate mass into a hill at x in [6.0, 7.5].
    t = init.copy().astype(np.float64)
    src = ((x > 4.5) & (x < 6.0)) | ((x > 7.5) & (x < 9.5))
    hill = (x > 6.0) & (x < 7.5)
    moved = t[src].sum() * 0.6
    t[src] *= 0.4
    t[hill] += moved / hill.sum()
    targets["central berm"] = _renorm(t)

    # 3. Flatten: level the workspace region toward its own mean.
    t = init.copy().astype(np.float64)
    ws = (x > 5.0) & (x < 9.5)
    t[ws] = t[ws].mean()
    targets["flatten workspace"] = _renorm(t)

    return targets


def main():
    args = parse_args()
    run_dir = ROOT / "runs" / args.run_name
    ckpt_path = run_dir / args.ckpt
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    assert getattr(cfg, "phi_input", "hist") == "hist", "need soil-phi ckpt"
    assert getattr(cfg, "measure", "1d") == "1d", "demo assumes 1d measure"

    env = SkillDiscoveryEnv(hist_bins=cfg.hist_bins,
                            max_episode_steps=args.steps,
                            randomize_terrain=True,
                            render_mode="human" if args.live else "rgb_array")
    trainer = TMSDTrainer(cfg, env.grid)
    trainer.load(str(ckpt_path))
    dev = trainer.device

    obs0, hist0, _ = env.reset(seed=args.seed)
    targets = make_targets(hist0.copy(), env.grid)

    results = {}
    for name, tgt in targets.items():
        obs, hist, _ = env.reset(seed=args.seed)
        tgt_t = torch.as_tensor(tgt, device=dev).unsqueeze(0)
        w2s = [w2_1d_exact(hist, tgt, env.grid)]
        if args.live:
            env.render()
            r = env.env._renderer
            r.pygame.display.set_caption(f"zero-shot shaping: {name}")
            heights = env.hist_to_heights(tgt)
            r.overlay_polyline = list(zip(env.grid, heights))
            r.overlay_label = f"target: {name}"
        frames = [env.render()]
        z = None
        for t in range(args.steps):
            if t % args.replan_every == 0:
                with torch.no_grad():
                    h = torch.as_tensor(hist, device=dev).unsqueeze(0)
                    dz = (trainer.phi(tgt_t) - trainer.phi(h)).squeeze(0).cpu().numpy()
                n = np.linalg.norm(dz)
                z = (dz / n if n > 1e-8 else trainer.sample_skill()).astype(np.float32)
            obs, hist, term, trunc, _ = env.step(trainer.act(obs, z, deterministic=True))
            if (t + 1) % args.replan_every == 0:
                w2s.append(w2_1d_exact(hist, tgt, env.grid))
            if t == args.steps // 2:
                frames.append(env.render())
            if term or trunc:
                break
        frames.append(env.render())
        results[name] = {"target": tgt, "achieved": hist.copy(),
                         "w2s": w2s, "frames": frames}
        print(f"{name:28s} W2 to goal {w2s[0]:.3f} -> {w2s[-1]:.3f} m "
              f"({1 - w2s[-1] / w2s[0]:+.0%})")
    env.close()
    if args.live:
        return  # live mode: no figure, the window was the deliverable

    # ── figure ───────────────────────────────────────────────────────
    n = len(results)
    fig = plt.figure(figsize=(15, 3.2 * n + 2.5))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[1.0] * n + [1.1])

    for i, (name, r) in enumerate(results.items()):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(env.grid, hist0, "k--", lw=1.5, label="initial")
        ax.plot(env.grid, r["target"], "r-", lw=2, label="target")
        ax.plot(env.grid, r["achieved"], "b-", lw=2, label="achieved")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x (m)")
        if i == 0:
            ax.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(r["frames"][1])
        ax2.set_title(f"{name} — mid", fontsize=9)
        ax2.axis("off")
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(r["frames"][2])
        ax3.set_title(f"{name} — end", fontsize=9)
        ax3.axis("off")

    axc = fig.add_subplot(gs[n, :])
    for name, r in results.items():
        steps = np.arange(len(r["w2s"])) * args.replan_every
        axc.plot(steps, r["w2s"], marker="o", ms=3, label=name)
    axc.set_xlabel("environment steps")
    axc.set_ylabel("W2 to target (m)")
    axc.set_title("Zero-shot contraction toward commanded profiles")
    axc.legend(fontsize=9)
    axc.grid(alpha=0.3)

    fig.suptitle(f"Zero-shot soil shaping — {args.run_name}", fontsize=13)
    fig.tight_layout()
    out = run_dir / "demo_shaping.png"
    fig.savefig(out, dpi=140)
    print(f"figure saved: {out}")

    with open(run_dir / "demo_shaping.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "step", "w2_to_goal"])
        for name, r in results.items():
            for k, v in enumerate(r["w2s"]):
                w.writerow([name, k * args.replan_every, v])


if __name__ == "__main__":
    main()
