"""
Zero-shot goal-reaching + metric-distortion evaluation.

Compares trained checkpoints (one per ground metric) on steering the
soil toward target profiles *without any further training*, using the
METRA-style zero-shot rule

    z_t = ( φ(μ*) − φ(μ_t) ) / ‖ φ(μ*) − φ(μ_t) ‖ ,

re-computed every ``--replan-every`` steps. If φ is metrically faithful
to physical transport (the W₂-training hypothesis), pointing z at the
goal's latent should move soil toward the goal; a diverse-but-warped φ
should steer worse. All methods are judged by physical W₂ to target.

Target pool: terminal soil states from every method's own skill
rollouts (displacement > 0.1 m from start), so each target is reachable
by construction and the pool is method-symmetric.

Also reports metric distortion per method: correlation between latent
distance ‖φ(a) − φ(b)‖ and physical W₂(a, b) over visited state pairs.

Outputs runs/goal_eval/(results.csv | distortion.csv | summary printed).

Usage:
    python scripts/eval_goal_reaching.py
    python scripts/eval_goal_reaching.py --runs tmsd_w2_rt_d4 abl_euclid_rt_d4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmsd.metrics import w2_1d, w2_1d_exact
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+",
                   default=["tmsd_w2_rt_d4", "abl_euclid_rt_d4",
                            "abl_temporal_rt_d4"])
    p.add_argument("--ckpt", type=str, default="ckpt_latest.pt")
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--replan-every", type=int, default=10)
    p.add_argument("--n-skills-pool", type=int, default=8)
    p.add_argument("--min-target-disp", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--randomize-terrain", action="store_true", default=True)
    return p.parse_args()


def load_trainer(run_name: str, ckpt: str, grid) -> TMSDTrainer:
    path = ROOT / "runs" / run_name / ckpt
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    if getattr(cfg, "phi_input", "hist") == "obs":
        raise ValueError(f"{run_name}: full-state φ cannot express soil-only "
                         "goals — excluded from zero-shot goal eval by design")
    if getattr(cfg, "measure", "1d") != "1d":
        raise ValueError(f"{run_name}: goal eval currently supports 1d-measure "
                         "checkpoints only (2d steering eval is a follow-up)")
    tr = TMSDTrainer(cfg, grid)
    tr.load(str(path))
    return tr


def skill_directions(n: int, dim: int, rng) -> np.ndarray:
    v = rng.normal(size=(n, dim))
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def rollout_skill(env, trainer, z, seed, steps) -> np.ndarray:
    obs, hist, _ = env.reset(seed=seed)
    for _ in range(steps):
        obs, hist, term, trunc, _ = env.step(trainer.act(obs, z, deterministic=True))
        if term or trunc:
            break
    return hist


def steer_to_goal(env, trainer, target_hist, seed, steps, replan_every):
    """Zero-shot goal-conditioned rollout. Returns (per-interval W2 list,
    visited hists)."""
    dev = trainer.device
    tgt = torch.as_tensor(target_hist, dtype=torch.float32, device=dev).unsqueeze(0)
    obs, hist, _ = env.reset(seed=seed)
    visited = [hist.copy()]
    w2s = [w2_1d_exact(hist, target_hist, env.grid)]
    z = None
    for t in range(steps):
        if t % replan_every == 0:
            with torch.no_grad():
                h = torch.as_tensor(hist, dtype=torch.float32,
                                    device=dev).unsqueeze(0)
                dz = (trainer.phi(tgt) - trainer.phi(h)).squeeze(0).cpu().numpy()
            n = np.linalg.norm(dz)
            z = (dz / n if n > 1e-8 else trainer.sample_skill()).astype(np.float32)
        obs, hist, term, trunc, _ = env.step(trainer.act(obs, z, deterministic=True))
        if (t + 1) % replan_every == 0:
            w2s.append(w2_1d_exact(hist, target_hist, env.grid))
            visited.append(hist.copy())
        if term or trunc:
            break
    w2s.append(w2_1d_exact(hist, target_hist, env.grid))
    visited.append(hist.copy())
    return w2s, visited


def distortion_stats(trainer, hists: np.ndarray, grid, n_pairs=2000, rng=None):
    """Correlation of ‖Δφ‖ vs physical W₂ over random visited-state pairs."""
    dev = trainer.device
    idx = rng.integers(0, len(hists), size=(n_pairs, 2))
    a = torch.as_tensor(hists[idx[:, 0]], dtype=torch.float32, device=dev)
    b = torch.as_tensor(hists[idx[:, 1]], dtype=torch.float32, device=dev)
    with torch.no_grad():
        dphi = torch.linalg.vector_norm(trainer.phi(a) - trainer.phi(b),
                                        dim=-1).cpu().numpy()
        g = torch.as_tensor(grid, dtype=torch.float32, device=dev)
        dw2 = w2_1d(a, b, g, n_quantiles=2048).cpu().numpy()
    keep = dw2 > 1e-6
    dphi, dw2 = dphi[keep], dw2[keep]
    pearson = float(np.corrcoef(dphi, dw2)[0, 1])
    rk = lambda x: np.argsort(np.argsort(x))
    spearman = float(np.corrcoef(rk(dphi), rk(dw2))[0, 1])
    ratio = dphi / dw2
    return pearson, spearman, float(np.std(np.log(ratio + 1e-12)))


def main():
    args = parse_args()
    out_dir = ROOT / "runs" / "goal_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env = SkillDiscoveryEnv(max_episode_steps=args.episode_steps,
                            randomize_terrain=args.randomize_terrain)
    trainers = {name: load_trainer(name, args.ckpt, env.grid)
                for name in args.runs}

    # ── build the shared, method-symmetric target pool ──────────────
    obs0, hist0, _ = env.reset(seed=args.seed)
    targets = []
    for name, tr in trainers.items():
        dirs = skill_directions(args.n_skills_pool, tr.cfg.skill_dim, rng)
        for z in dirs:
            h = rollout_skill(env, tr, z, args.seed, args.episode_steps)
            disp = w2_1d_exact(hist0, h, env.grid)
            if disp > args.min_target_disp:
                targets.append((name, h, disp))
    print(f"target pool: {len(targets)} reachable targets "
          f"(displacement {args.min_target_disp}+ m) from {len(trainers)} methods\n")

    # ── steering eval ────────────────────────────────────────────────
    rows = []
    all_visited = {name: [] for name in trainers}
    for name, tr in trainers.items():
        for ti, (src, tgt_hist, tgt_disp) in enumerate(targets):
            w2s, visited = steer_to_goal(env, tr, tgt_hist, args.seed,
                                         args.episode_steps, args.replan_every)
            all_visited[name].extend(visited)
            diffs = np.diff(w2s)
            rows.append({
                "method": name, "target": ti, "target_src": src,
                "w2_initial": w2s[0], "w2_final": w2s[-1],
                "w2_best": min(w2s),
                "improvement": 1.0 - w2s[-1] / max(w2s[0], 1e-9),
                "monotone_frac": float((diffs <= 0).mean()),
            })
        sub = [r for r in rows if r["method"] == name]
        imp = np.mean([r["improvement"] for r in sub])
        fin = np.mean([r["w2_final"] for r in sub])
        mono = np.mean([r["monotone_frac"] for r in sub])
        print(f"{name:22s} improvement {imp:+.1%}  final W2 {fin:.3f} m  "
              f"monotone {mono:.0%}")

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ── metric distortion ────────────────────────────────────────────
    print("\nmetric distortion (|dphi| vs physical W2 over visited pairs):")
    drows = []
    for name, tr in trainers.items():
        hists = np.stack(all_visited[name])
        pear, spear, logstd = distortion_stats(tr, hists, env.grid, rng=rng)
        drows.append({"method": name, "pearson": pear, "spearman": spear,
                      "log_ratio_std": logstd})
        print(f"{name:22s} pearson {pear:.3f}  spearman {spear:.3f}  "
              f"log-distortion std {logstd:.3f}")
    with open(out_dir / "distortion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0]))
        w.writeheader()
        w.writerows(drows)

    env.close()
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
