"""
Excavate-and-dump demo: command the excavator to remove soil from a dig
zone and deposit it in a dump zone, using only unsupervised skills +
zero-shot latent steering (z = phi(target) - phi(current), replanned
every few steps). No task reward, no planner, no goal-conditioned
training.

The target profile is built from the CLI zones: a chosen fraction of the
dig zone's mass is moved into the dump zone (mass-conserving). Success is
reported as per-zone mass accounting plus W2-to-target.

Note: the arm is anchored at x=2 with ~9.4 m reach — zones outside
x ~ [4.5, 9.5] are physically unreachable and are rejected.

Usage:
    python scripts/demo_excavate_dump.py --dig 7 9 --dump 4.5 6.5 --live
    python scripts/demo_excavate_dump.py --dig 5 6.5 --dump 7.5 9 --frac 0.7
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

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = (4.5, 9.5)   # arm's effective reach on the soil bed (m)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, default="tmsd_w2_rt_d4")
    p.add_argument("--ckpt", type=str, default="ckpt_latest.pt")
    p.add_argument("--dig", type=float, nargs=2, required=True,
                   metavar=("X0", "X1"), help="dig zone bounds (m)")
    p.add_argument("--dump", type=float, nargs=2, required=True,
                   metavar=("X0", "X1"), help="dump zone bounds (m)")
    p.add_argument("--frac", type=float, default=0.8,
                   help="fraction of dig-zone mass to relocate")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--replan-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--live", action="store_true", help="render live window")
    return p.parse_args()


def check_zone(name, lo, hi):
    if not (WORKSPACE[0] - 0.51 <= lo < hi <= WORKSPACE[1] + 0.51):
        raise SystemExit(
            f"{name} zone [{lo}, {hi}] outside arm workspace "
            f"~[{WORKSPACE[0]}, {WORKSPACE[1]}] m — physically unreachable.")


def zone_mass(hist, grid, lo, hi) -> float:
    m = (grid >= lo) & (grid <= hi)
    return float(hist[m].sum())


def make_target(init, grid, dig, dump, frac) -> np.ndarray:
    t = init.astype(np.float64).copy()
    src = (grid >= dig[0]) & (grid <= dig[1])
    dst = (grid >= dump[0]) & (grid <= dump[1])
    moved = t[src].sum() * frac
    t[src] *= (1.0 - frac)
    t[dst] += moved / dst.sum()
    t = np.clip(t, 0, None)
    return (t / t.sum()).astype(np.float32)


def main():
    args = parse_args()
    check_zone("dig", *args.dig)
    check_zone("dump", *args.dump)

    ckpt_path = ROOT / "runs" / args.run_name / args.ckpt
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    assert getattr(cfg, "phi_input", "hist") == "hist"

    env = SkillDiscoveryEnv(hist_bins=cfg.hist_bins,
                            max_episode_steps=args.steps,
                            randomize_terrain=True,
                            render_mode="human" if args.live else None)
    trainer = TMSDTrainer(cfg, env.grid)
    trainer.load(str(ckpt_path))
    dev = trainer.device

    obs, hist, _ = env.reset(seed=args.seed)
    init = hist.copy()
    target = make_target(init, env.grid, args.dig, args.dump, args.frac)
    tgt_t = torch.as_tensor(target, device=dev).unsqueeze(0)

    if args.live:
        env.render()
        r = env.env._renderer
        r.pygame.display.set_caption(
            f"excavate [{args.dig[0]}-{args.dig[1]}m] -> "
            f"dump [{args.dump[0]}-{args.dump[1]}m]")
        heights = env.hist_to_heights(target)
        r.overlay_polyline = list(zip(env.grid, heights))
        r.overlay_label = "target profile (red)"

    dig0 = zone_mass(init, env.grid, *args.dig)
    dump0 = zone_mass(init, env.grid, *args.dump)
    w2_0 = w2_1d_exact(init, target, env.grid)
    print(f"initial: dig-zone mass {dig0:.1%}, dump-zone mass {dump0:.1%}, "
          f"W2 to target {w2_0:.3f} m")

    z = None
    for t in range(args.steps):
        if t % args.replan_every == 0:
            with torch.no_grad():
                h = torch.as_tensor(hist, device=dev).unsqueeze(0)
                dz = (trainer.phi(tgt_t) - trainer.phi(h)).squeeze(0).cpu().numpy()
            n = np.linalg.norm(dz)
            z = (dz / n if n > 1e-8 else trainer.sample_skill()).astype(np.float32)
        obs, hist, term, trunc, _ = env.step(trainer.act(obs, z, deterministic=True))
        if (t + 1) % 100 == 0:
            print(f"  step {t + 1:4d}: dig {zone_mass(hist, env.grid, *args.dig):.1%} "
                  f"dump {zone_mass(hist, env.grid, *args.dump):.1%} "
                  f"W2 {w2_1d_exact(hist, target, env.grid):.3f}")
        if term or trunc:
            break
    env.close()

    dig1 = zone_mass(hist, env.grid, *args.dig)
    dump1 = zone_mass(hist, env.grid, *args.dump)
    w2_1 = w2_1d_exact(hist, target, env.grid)
    tgt_dig = zone_mass(target, env.grid, *args.dig)
    tgt_dump = zone_mass(target, env.grid, *args.dump)
    print(f"\nresult:")
    print(f"  dig zone : {dig0:.1%} -> {dig1:.1%}  (target {tgt_dig:.1%})  "
          f"removed {(dig0 - dig1) / max(dig0, 1e-9):.0%} of its soil")
    print(f"  dump zone: {dump0:.1%} -> {dump1:.1%}  (target {tgt_dump:.1%})  "
          f"gained {dump1 - dump0:+.1%} of total mass")
    print(f"  W2 to target: {w2_0:.3f} -> {w2_1:.3f} m ({1 - w2_1 / w2_0:+.0%})")


if __name__ == "__main__":
    main()
