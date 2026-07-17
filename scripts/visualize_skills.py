"""
Visualize discovered skills from a TMSD checkpoint.

Rolls out K skill directions (evenly spaced on the circle for
skill_dim=2, random unit vectors otherwise), then saves to the run dir:

    skills_heightfields.png   terminal soil profile per skill vs initial
    skills_coverage.png       pairwise-W2 matrix between terminal soil states
    skills_latent.png         phi-trajectories in latent space, colored by skill
    coverage.csv              the raw pairwise-W2 numbers

Usage:
    python scripts/visualize_skills.py --run-name tmsd_w2 --n-skills 8
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, default="tmsd_w2")
    p.add_argument("--ckpt", type=str, default="ckpt_latest.pt")
    p.add_argument("--n-skills", type=int, default=8)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--randomize-terrain", action="store_true",
                   help="match training-time terrain randomization (same "
                        "seed still gives every skill the same terrain)")
    return p.parse_args()


def skill_directions(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    if dim == 2:
        angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
    v = rng.normal(size=(n, dim))
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def main():
    args = parse_args()
    run_dir = Path(__file__).resolve().parents[1] / "runs" / args.run_name
    ckpt_path = run_dir / args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = ckpt["cfg"]

    measure = getattr(cfg, "measure", "1d")
    env = SkillDiscoveryEnv(measure=measure,
                            hist_bins=cfg.hist_bins if measure == "1d" else 64,
                            max_episode_steps=args.episode_steps,
                            randomize_terrain=args.randomize_terrain)
    trainer = TMSDTrainer(cfg, env.grid)
    trainer.load(str(ckpt_path))

    rng = np.random.default_rng(args.seed)
    zs = skill_directions(args.n_skills, cfg.skill_dim, rng)

    phi_on_obs = getattr(cfg, "phi_input", "hist") == "obs"
    terminal_hists, initial_hist, phi_trajs = [], None, []
    for k, z in enumerate(zs):
        obs, hist, _ = env.reset(seed=args.seed)  # same seed: same initial pile
        if initial_hist is None:
            initial_hist = env.eval_hist_1d()     # 1D yardstick for plots
        traj = [obs.copy() if phi_on_obs else hist.copy()]
        for _ in range(args.episode_steps):
            action = trainer.act(obs, z, deterministic=True)
            obs, hist, terminated, truncated, _ = env.step(action)
            traj.append(obs.copy() if phi_on_obs else hist.copy())
            if terminated or truncated:
                break
        terminal_hists.append(env.eval_hist_1d())
        with torch.no_grad():
            phis = trainer.phi(torch.as_tensor(np.stack(traj), dtype=torch.float32,
                                               device=trainer.device))
        phi_trajs.append(phis.cpu().numpy())
        moved = w2_1d_exact(initial_hist, terminal_hists[-1], env.grid_eval)
        print(f"skill {k} z=({', '.join(f'{v:+.2f}' for v in z)}) "
              f"terminal W2 from start = {moved:.4f} m")
    env.close()

    # Pairwise W2 between terminal soil states.
    n = len(terminal_hists)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cov[i, j] = cov[j, i] = w2_1d_exact(terminal_hists[i],
                                                terminal_hists[j], env.grid_eval)
    with open(run_dir / "coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["skill_i", "skill_j", "w2"])
        for i in range(n):
            for j in range(n):
                w.writerow([i, j, cov[i, j]])
    off_diag = cov[~np.eye(n, dtype=bool)]
    print(f"pairwise terminal W2: mean={off_diag.mean():.4f} "
          f"min={off_diag.min():.4f} max={off_diag.max():.4f}")

    cmap = plt.get_cmap("hsv")
    colors = [cmap(k / n) for k in range(n)]

    # 1. Terminal soil profiles.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(env.grid_eval, initial_hist, "k--", lw=2, label="initial")
    for k in range(n):
        ax.plot(env.grid_eval, terminal_hists[k], color=colors[k], label=f"skill {k}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("soil mass fraction")
    ax.set_title("Terminal soil distribution per skill")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "skills_heightfields.png", dpi=150)

    # 2. Coverage matrix.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cov, cmap="viridis")
    fig.colorbar(im, ax=ax, label="W2 (m)")
    ax.set_title("Pairwise W2 between terminal soil states")
    ax.set_xlabel("skill")
    ax.set_ylabel("skill")
    fig.tight_layout()
    fig.savefig(run_dir / "skills_coverage.png", dpi=150)

    # 3. phi-space trajectories (first two latent dims).
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, phis in enumerate(phi_trajs):
        ax.plot(phis[:, 0], phis[:, 1], color=colors[k], label=f"skill {k}")
        ax.scatter(phis[-1, 0], phis[-1, 1], color=colors[k], marker="*", s=80)
    ax.set_xlabel('phi_1')
    ax.set_ylabel('phi_2')
    ax.set_title('phi(soil) trajectories per skill (* = terminal)')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "skills_latent.png", dpi=150)

    print(f"plots saved to {run_dir}")


if __name__ == "__main__":
    main()

