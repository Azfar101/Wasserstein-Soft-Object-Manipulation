"""
Train TMSD on the excavator DEM sim.

Usage:
    python scripts/train_tmsd.py --steps 500000 --run-name tmsd_w2
    python scripts/train_tmsd.py --metric euclidean --run-name ablation_l2

Outputs under runs/<run-name>/:
    log.csv          per-interval training stats
    episodes.csv     per-episode soil displacement + return
    ckpt_latest.pt / ckpt_<step>.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmsd.metrics import w2_1d_exact
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--run-name", type=str, default="tmsd_w2")
    p.add_argument("--metric", type=str, default="w2",
                   choices=["w2", "euclidean", "temporal", "sliced_w2"])
    p.add_argument("--measure", type=str, default="1d",
                   choices=["1d", "2d", "composite"],
                   help="2d = (x,y) mass grid; composite = ground/bucket/"
                        "airborne mass partition (carry-aware)")
    p.add_argument("--brush-ops", type=int, default=0,
                   help="random user-like sculpt edits per reset")
    p.add_argument("--persist-soil-prob", type=float, default=0.0,
                   help="prob of keeping soil across episode resets")
    p.add_argument("--arm-random-steps", type=int, default=0,
                   help="random actions at reset to vary arm pose")
    p.add_argument("--phi-input", type=str, default="hist",
                   choices=["hist", "obs"],
                   help="obs = full-state discriminator (baseline ablation)")
    p.add_argument("--objective", type=str, default="metra",
                   choices=["metra", "mi", "dads"],
                   help="mi = continuous-DIAYN; dads = skill-dynamics")
    p.add_argument("--reward-scale", type=float, default=None,
                   help="default: 100 for metra, 1 for mi/dads")
    p.add_argument("--skill-dim", type=int, default=2)
    p.add_argument("--hist-bins", type=int, default=64)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--ckpt-every", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--randomize-terrain", action="store_true",
                   help="random height envelope per episode")
    p.add_argument("--force-cpu-sim", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(__file__).resolve().parents[1] / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.measure == "composite" and args.metric in ("w2", "sliced_w2"):
        raise SystemExit("composite measure has heterogeneous channels; "
                         "use --metric euclidean or temporal")
    env = SkillDiscoveryEnv(measure=args.measure, hist_bins=args.hist_bins,
                            max_episode_steps=args.episode_steps,
                            randomize_terrain=args.randomize_terrain,
                            brush_ops=args.brush_ops,
                            persist_soil_prob=args.persist_soil_prob,
                            arm_random_steps=args.arm_random_steps,
                            force_cpu=args.force_cpu_sim)
    reward_scale = (args.reward_scale if args.reward_scale is not None
                    else (1.0 if args.objective in ("mi", "dads") else 100.0))
    cfg = TMSDConfig(obs_dim=env.obs_dim, hist_bins=env.measure_dim,
                     act_dim=env.act_dim, skill_dim=args.skill_dim,
                     metric=args.metric, phi_input=args.phi_input,
                     objective=args.objective, reward_scale=reward_scale,
                     measure=args.measure, seed=args.seed,
                     # 2d histograms are 18× larger; bound replay RAM.
                     buffer_capacity=150_000 if args.measure == "2d" else 300_000)
    trainer = TMSDTrainer(cfg, env.grid)
    print(f"run={args.run_name} metric={args.metric} skill_dim={args.skill_dim} "
          f"device={cfg.device} sim_backend={'cpu' if args.force_cpu_sim else 'auto'}")

    log_path = run_dir / "log.csv"
    ep_path = run_dir / "episodes.csv"
    log_file = open(log_path, "a", newline="")
    ep_file = open(ep_path, "a", newline="")
    log_writer = ep_writer = None  # created lazily once headers are known

    obs, hist, _ = env.reset(seed=args.seed)
    eval0 = env.eval_hist_1d()
    z = trainer.sample_skill()
    ep_return, ep_len, ep_count = 0.0, 0, 0
    stats_acc: dict[str, list] = {}
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        if step <= args.warmup_steps:
            action = np.random.default_rng(args.seed + step).uniform(-1, 1, env.act_dim)
        else:
            action = trainer.act(obs, z)

        next_obs, next_hist, terminated, truncated, _ = env.step(action)
        trainer.buffer.push(obs, hist, action, next_obs, next_hist, z, terminated)

        pin, pin_next = ((obs, next_obs) if args.phi_input == "obs"
                         else (hist, next_hist))
        with torch.no_grad():
            r = trainer.intrinsic_reward(
                torch.as_tensor(pin, device=trainer.device).unsqueeze(0),
                torch.as_tensor(pin_next, device=trainer.device).unsqueeze(0),
                torch.as_tensor(z, device=trainer.device).unsqueeze(0)).item()
        ep_return += r
        ep_len += 1
        obs, hist = next_obs, next_hist

        if terminated or truncated:
            soil_moved = w2_1d_exact(eval0, env.eval_hist_1d(), env.grid_eval)
            ep_count += 1
            row = {"episode": ep_count, "step": step, "return": ep_return,
                   "len": ep_len, "terminal_w2_from_start": soil_moved,
                   "z": " ".join(f"{v:+.3f}" for v in z)}
            if ep_writer is None:
                ep_writer = csv.DictWriter(ep_file, fieldnames=list(row))
                if ep_path.stat().st_size == 0:
                    ep_writer.writeheader()
            ep_writer.writerow(row)
            ep_file.flush()
            obs, hist, _ = env.reset()
            eval0 = env.eval_hist_1d()
            z = trainer.sample_skill()
            ep_return, ep_len = 0.0, 0

        if step > args.warmup_steps:
            for _ in range(args.updates_per_step):
                for k, v in trainer.update().items():
                    stats_acc.setdefault(k, []).append(v)

        if step % args.log_every == 0:
            sps = step / (time.perf_counter() - t_start)
            row = {"step": step, "sps": round(sps, 1), "episodes": ep_count}
            has_stats = bool(stats_acc)
            row.update({k: float(np.mean(v)) for k, v in sorted(stats_acc.items())})
            stats_acc.clear()
            # CSV schema includes trainer stats; skip rows logged before the
            # first update (warmup) so the header is complete when created.
            if has_stats:
                if log_writer is None:
                    log_writer = csv.DictWriter(log_file, fieldnames=list(row))
                    if log_path.stat().st_size == 0:
                        log_writer.writeheader()
                log_writer.writerow(row)
                log_file.flush()
            printable = {k: (f"{v:.4g}" if isinstance(v, float) else v)
                         for k, v in row.items()}
            print(printable)

        if step % args.ckpt_every == 0:
            trainer.save(str(run_dir / f"ckpt_{step}.pt"))
            trainer.save(str(run_dir / "ckpt_latest.pt"))

    trainer.save(str(run_dir / "ckpt_latest.pt"))
    log_file.close()
    ep_file.close()
    env.close()
    print(f"done: {args.steps} steps, {ep_count} episodes -> {run_dir}")


if __name__ == "__main__":
    main()
