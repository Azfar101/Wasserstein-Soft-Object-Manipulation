"""
Live representation viewer: the sim window plus a second window showing,
in real time, what the method's representations are doing --
the soil measure mu (what the objective sees) and the latent trajectory
phi(mu) (where the state is moving on the learned transport map).

Cycles through skill directions like watch_skills, one episode each.
Close either window / Ctrl+C to stop.

    python scripts/watch_representation.py [--run-name tmsd_w2_rt_d4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2a78d6", "#e34948", "#1baf7a", "#eda100",
          "#4a3aa7", "#e87ba4", "#eb6834", "#008300"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, default="tmsd_w2_rt_d4")
    p.add_argument("--n-skills", type=int, default=6)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = ROOT / "runs" / args.run_name / "ckpt_latest.pt"
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]

    env = SkillDiscoveryEnv(hist_bins=cfg.hist_bins,
                            max_episode_steps=args.episode_steps,
                            randomize_terrain=True, render_mode="human")
    tr = TMSDTrainer(cfg, env.grid)
    tr.load(str(ckpt))

    rng = np.random.default_rng(args.seed)
    v = rng.normal(size=(args.n_skills, cfg.skill_dim))
    zs = (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)

    plt.ion()
    fig, (ax_mu, ax_phi) = plt.subplots(1, 2, figsize=(10, 4))
    fig.canvas.manager.set_window_title("live representation: mu and phi")
    fig.suptitle("What the method sees, live", fontweight="bold")

    def phi_of(h):
        with torch.no_grad():
            return tr.phi(torch.as_tensor(h, dtype=torch.float32,
                                          device=tr.device).unsqueeze(0)
                          ).cpu().numpy()[0]

    try:
        while plt.fignum_exists(fig.number):
            for k, z in enumerate(zs):
                col = COLORS[k % len(COLORS)]
                obs, hist, _ = env.reset(seed=args.seed)
                env.render()
                env.env._renderer.pygame.display.set_caption(
                    f"skill {k + 1}/{len(zs)}")
                hist0 = hist.copy()
                phis = [phi_of(hist)]
                for t in range(args.episode_steps):
                    obs, hist, term, trunc, _ = env.step(
                        tr.act(obs, z, deterministic=True))
                    # renderer quit?
                    r = env.env._renderer
                    for ev in r.pygame.event.get(r.pygame.QUIT):
                        raise KeyboardInterrupt
                    if t % 5 == 0 or term or trunc:
                        phis.append(phi_of(hist))
                        P = np.stack(phis)
                        ax_mu.clear()
                        ax_mu.bar(env.grid, hist0, width=0.2, color="0.75",
                                  label="initial")
                        ax_mu.bar(env.grid, hist, width=0.2, color=col,
                                  alpha=0.7, label="now")
                        ax_mu.set_title(
                            "$\\mu$: soil mass distribution (objective's view)")
                        ax_mu.set_xlabel("x (m)")
                        ax_mu.legend(fontsize=7, loc="upper right")
                        ax_phi.clear()
                        ax_phi.plot(P[:, 0], P[:, 1], color=col, lw=2)
                        ax_phi.scatter(P[0, 0], P[0, 1], color="k", s=30,
                                       zorder=4)
                        ax_phi.scatter(P[-1, 0], P[-1, 1], color=col,
                                       marker="*", s=140, zorder=4)
                        ax_phi.set_title(
                            f"$\\phi(\\mu)$ trajectory, skill {k + 1} "
                            "(dot=start, star=now)")
                        ax_phi.set_xlabel("$\\phi_1$")
                        ax_phi.set_ylabel("$\\phi_2$")
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    if term or trunc:
                        break
                    if not plt.fignum_exists(fig.number):
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plt.close("all")
        print("stopped")


if __name__ == "__main__":
    main()
