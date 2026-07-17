"""
Live viewer for trained TMSD skills.

Opens the sim's pygame window and rolls out one skill after another with
the deterministic policy, printing per-skill soil-transport stats to the
console. Window caption shows the active skill. Close the window (or
Ctrl+C) to stop early.

Usage:
    python scripts/watch_skills.py --run-name tmsd_w2_s0
    python scripts/watch_skills.py --run-name tmsd_w2_s0 --n-skills 8 --loop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmsd.metrics import w2_1d_exact
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, default="tmsd_w2_s0")
    p.add_argument("--ckpt", type=str, default="ckpt_latest.pt")
    p.add_argument("--n-skills", type=int, default=8)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--loop", action="store_true",
                   help="cycle through the skills forever")
    p.add_argument("--randomize-terrain", action="store_true",
                   help="match training-time terrain randomization")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of deterministic mean")
    return p.parse_args()


def skill_directions(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    if dim == 2:
        angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
    v = rng.normal(size=(n, dim))
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def window_alive(env) -> bool:
    """False once the user closes the pygame window."""
    r = env.env._renderer
    if r is None:
        return True
    for event in r.pygame.event.get(r.pygame.QUIT):
        return False
    return True


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
                            randomize_terrain=args.randomize_terrain,
                            render_mode="human")
    trainer = TMSDTrainer(cfg, env.grid)
    trainer.load(str(ckpt_path))

    rng = np.random.default_rng(args.seed)
    zs = skill_directions(args.n_skills, cfg.skill_dim, rng)
    print(f"ckpt: {ckpt_path}\nskills: {args.n_skills} | "
          f"episode: {args.episode_steps} steps | "
          f"{'stochastic' if args.stochastic else 'deterministic'} policy\n"
          f"close the window or Ctrl+C to stop\n")

    try:
        cycle = 0
        while True:
            for k, z in enumerate(zs):
                obs, hist, _ = env.reset(seed=args.seed)  # same pile every skill
                eval0 = env.eval_hist_1d()
                env.render()  # ensure the window exists before setting caption
                zstr = ", ".join(f"{v:+.2f}" for v in z)
                caption = f"TMSD skill {k + 1}/{len(zs)}  z=({zstr})"
                env.env._renderer.pygame.display.set_caption(caption)
                for _ in range(args.episode_steps):
                    action = trainer.act(obs, z, deterministic=not args.stochastic)
                    obs, hist, terminated, truncated, _ = env.step(action)
                    if not window_alive(env):
                        raise KeyboardInterrupt
                    if terminated or truncated:
                        break
                ev = env.eval_hist_1d()
                moved = w2_1d_exact(eval0, ev, env.grid_eval)
                shift = float(np.sum(env.grid_eval * (ev - eval0)))  # signed CM shift
                print(f"skill {k} z=({zstr})  W2 moved={moved:.4f} m  "
                      f"soil CM shift={shift:+.4f} m")
            cycle += 1
            if not args.loop:
                break
            print(f"--- cycle {cycle} done, looping ---\n")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()


if __name__ == "__main__":
    main()
