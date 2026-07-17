"""
Train the goal-conditioned shaping controller (see tmsd/gc_trainer.py).

Task curriculum per episode (matches the interactive demo's jobs):
  40%  RESTORE  goal = settled profile, then user-like mess is applied —
                the episode must undo it (the record/restore demo, literally)
  30%  ZONES    goal = random dig->dump zone edit of the current profile
                (the excavate-and-dump demo, literally)
  30%  BANK     goal = a previously *achieved* terminal profile (reachable
                by construction; keeps goal diversity high)

Usage:
    python scripts/train_goal_conditioned.py --steps 600000 --run-name gc_s0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmsd.gc_trainer import GCTrainer, GCConfig
from tmsd.metrics import w2_1d_exact
from tmsd.wrappers import SkillDiscoveryEnv

WORKSPACE = (4.5, 9.5)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=600_000)
    p.add_argument("--run-name", type=str, default="gc_s0")
    p.add_argument("--episode-steps", type=int, default=300)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--log-every", type=int, default=2_000)
    p.add_argument("--ckpt-every", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def zone_goal(hist, grid, rng):
    lo = rng.uniform(WORKSPACE[0], WORKSPACE[1] - 1.5)
    dig = (lo, lo + rng.uniform(0.8, 2.0))
    # dump machine-side or far-side of the dig span, inside the workspace
    if rng.random() < 0.5 and dig[0] - WORKSPACE[0] > 1.0:
        dump = (WORKSPACE[0], dig[0] - 0.2)
    else:
        dump = (min(dig[1] + 0.2, WORKSPACE[1] - 0.5), WORKSPACE[1])
    t = hist.astype(np.float64).copy()
    src = (grid >= dig[0]) & (grid <= dig[1])
    dst = (grid >= dump[0]) & (grid <= dump[1])
    if not src.any() or not dst.any():
        return hist.copy()
    frac = rng.uniform(0.5, 0.9)
    moved = t[src].sum() * frac
    t[src] *= (1.0 - frac)
    t[dst] += moved / dst.sum()
    return (np.clip(t, 0, None) / t.sum()).astype(np.float32)


def main():
    args = parse_args()
    run_dir = Path(__file__).resolve().parents[1] / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env = SkillDiscoveryEnv(hist_bins=64,
                            max_episode_steps=args.episode_steps,
                            randomize_terrain=True,
                            brush_ops=2,
                            persist_soil_prob=0.35,
                            arm_random_steps=10)
    cfg = GCConfig(obs_dim=env.obs_dim, goal_dim=64, act_dim=env.act_dim,
                   seed=args.seed)
    trainer = GCTrainer(cfg, env.grid_eval)
    print(f"run={args.run_name} device={cfg.device}")

    goal_bank: list[np.ndarray] = []
    log_f = open(run_dir / "log.csv", "a", newline="")
    ep_f = open(run_dir / "episodes.csv", "a", newline="")
    log_w = ep_w = None
    stats_acc: dict[str, list] = {}
    t0 = time.perf_counter()

    step = 0
    ep_count = 0
    while step < args.steps:
        # ── episode setup: draw a task ─────────────────────────────
        obs, _, _ = env.reset()
        u = rng.random()
        if u < 0.4:
            task = "restore"
            goal = env.eval_hist_1d().copy()
            env.apply_user_mess(int(rng.integers(3, 9)))
            obs = env.env._build_obs()
        elif u < 0.7:
            task = "zones"
            goal = zone_goal(env.eval_hist_1d(), env.grid_eval, rng)
        else:
            task = "bank"
            goal = (goal_bank[rng.integers(len(goal_bank))]
                    if goal_bank else env.eval_hist_1d().copy())

        w2_start = w2_1d_exact(env.eval_hist_1d(), goal, env.grid_eval)
        hist = env.eval_hist_1d()
        transitions = []
        for _ in range(args.episode_steps):
            if step <= args.warmup_steps:
                action = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
            else:
                action = trainer.act(obs, goal)
            next_obs, _, term, trunc, _ = env.step(action)
            next_hist = env.eval_hist_1d()
            transitions.append((obs, hist, action, next_obs, next_hist))
            obs, hist = next_obs, next_hist
            step += 1
            if step > args.warmup_steps:
                for k, v in trainer.update().items():
                    stats_acc.setdefault(k, []).append(v)
            if step % args.log_every == 0:
                row = {"step": step,
                       "sps": round(step / (time.perf_counter() - t0), 1),
                       "episodes": ep_count}
                row.update({k: float(np.mean(v))
                            for k, v in sorted(stats_acc.items())})
                stats_acc.clear()
                if stats_acc is not None and len(row) > 3:
                    if log_w is None:
                        log_w = csv.DictWriter(log_f, fieldnames=list(row))
                        if (run_dir / "log.csv").stat().st_size == 0:
                            log_w.writeheader()
                    log_w.writerow(row)
                    log_f.flush()
                print({k: (f"{v:.4g}" if isinstance(v, float) else v)
                       for k, v in row.items()})
            if step % args.ckpt_every == 0:
                trainer.save(str(run_dir / "ckpt_latest.pt"))
                trainer.save(str(run_dir / f"ckpt_{step}.pt"))
            if term or trunc:
                break

        trainer.push_episode(transitions, goal)
        w2_end = w2_1d_exact(env.eval_hist_1d(), goal, env.grid_eval)
        if len(goal_bank) < 800:
            goal_bank.append(env.eval_hist_1d().copy())
        else:
            goal_bank[rng.integers(800)] = env.eval_hist_1d().copy()
        ep_count += 1
        row = {"episode": ep_count, "step": step, "task": task,
               "w2_start": w2_start, "w2_end": w2_end,
               "improvement": (1.0 - w2_end / w2_start) if w2_start > 1e-6 else 0.0}
        if ep_w is None:
            ep_w = csv.DictWriter(ep_f, fieldnames=list(row))
            if (run_dir / "episodes.csv").stat().st_size == 0:
                ep_w.writeheader()
        ep_w.writerow(row)
        ep_f.flush()

    trainer.save(str(run_dir / "ckpt_latest.pt"))
    log_f.close()
    ep_f.close()
    env.close()
    print(f"done: {args.steps} steps, {ep_count} episodes -> {run_dir}")


if __name__ == "__main__":
    main()
