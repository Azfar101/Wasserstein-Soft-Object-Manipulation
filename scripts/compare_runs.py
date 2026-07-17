"""
Frozen evaluation protocol for cross-run comparison (two yardsticks).

Written and fixed BEFORE the overnight-batch results were seen, so the
ruler cannot be chosen after the scores: every run — regardless of its
training measure or metric — is scored on BOTH:

  1D coverage / displacement:  W2 between 64-bin x-marginals
     (horizontal transport; blind to vertical structure)
  2D coverage / displacement:  sliced-W2 (128 fixed projections, seed 0)
     between 48x24 (x, y) mass histograms (sees vertical structure)

Protocol: for each run, roll out the SAME 8 skill directions (seed 123,
skill_dim=4) on the SAME 3 randomized terrains (reset seeds 123/456/789),
deterministic policy, 200 steps. Coverage = mean pairwise distance
between terminal states within a terrain, averaged over terrains.
Displacement = mean distance from the terrain's initial state.

Usage:
    python scripts/compare_runs.py --runs tmsd_w2_rt_d4 abl_euclid_rt_d4 ...
    python scripts/compare_runs.py --all-batch
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmsd.measures import (soil_mass_histogram, soil_mass_histogram_2d,
                           bin_centers, bin_centers_2d)
from tmsd.metrics import w2_1d_exact, SlicedW2
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

ROOT = Path(__file__).resolve().parents[1]

EVAL_TERRAIN_SEEDS = (123, 456, 789)
SKILL_SEED = 123
N_SKILLS = 8
EPISODE_STEPS = 200
BX, BY = 48, 24

BATCH_RUNS = [
    # 1D-measure family (seed replicates)
    "tmsd_w2_rt_d4", "tmsd_w2_rt_d4_s1", "tmsd_w2_rt_d4_s2",
    "abl_euclid_rt_d4", "abl_euclid_rt_d4_s1", "abl_euclid_rt_d4_s2",
    "abl_temporal_rt_d4", "abl_temporal_rt_d4_s1", "abl_temporal_rt_d4_s2",
    "abl_fullstate_temporal", "abl_fullstate_temporal_s1", "abl_fullstate_temporal_s2",
    # 2D-measure trio
    "tmsd_sw2_2d_s0", "abl_euclid_2d_s0", "abl_temporal_2d_s0",
    # MI 2x2 completion (continuous DIAYN)
    "mi_soil_s0", "mi_soil_s1", "mi_soil_s2",
    "mi_full_s0", "mi_full_s1", "mi_full_s2",
    # DADS 2x2 (skill dynamics)
    "dads_soil_s0", "dads_soil_s1", "dads_full_s0", "dads_full_s1",
    # stage-2 retrains (composite measure + state randomization)
    "stage2_comp_s0", "stage2_comp_s1", "stage2_1d_s0",
    # 500k fixed-terrain pilot
    "tmsd_w2_s0",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=None)
    p.add_argument("--all-batch", action="store_true")
    p.add_argument("--ckpt", type=str, default="ckpt_latest.pt")
    return p.parse_args()


def load(run_name: str, ckpt: str, grid):
    path = ROOT / "runs" / run_name / ckpt
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    tr = TMSDTrainer(cfg, grid)
    tr.load(str(path))
    return tr, cfg


def eval_run(run_name: str, ckpt: str, sw2_eval, rng_dirs) -> dict:
    # Probe the ckpt for its measure to build a matching env.
    saved = torch.load(ROOT / "runs" / run_name / ckpt,
                       map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    measure = getattr(cfg, "measure", "1d")
    env = SkillDiscoveryEnv(measure=measure,
                            hist_bins=cfg.hist_bins if measure == "1d" else 64,
                            max_episode_steps=EPISODE_STEPS,
                            randomize_terrain=True)
    trainer = TMSDTrainer(cfg, env.grid)
    trainer.load(str(ROOT / "runs" / run_name / ckpt))
    grid1d = bin_centers(env.env.cfg, 64)

    # Same seed, dimension matched to the checkpoint (protocol-fixed).
    rng = np.random.default_rng(SKILL_SEED)
    v = rng.normal(size=(N_SKILLS, cfg.skill_dim))
    zs = (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)
    cov1, cov2, disp1, disp2 = [], [], [], []
    for ts in EVAL_TERRAIN_SEEDS:
        term1, term2 = [], []
        init1 = init2 = None
        for z in zs:
            obs, hist, _ = env.reset(seed=ts)
            if init1 is None:
                init1 = soil_mass_histogram(env.env.ps, env.env.cfg, 64)
                init2 = soil_mass_histogram_2d(env.env.ps, env.env.cfg, BX, BY)
            for _ in range(EPISODE_STEPS):
                obs, hist, term, trunc, _ = env.step(
                    trainer.act(obs, z, deterministic=True))
                if term or trunc:
                    break
            term1.append(soil_mass_histogram(env.env.ps, env.env.cfg, 64))
            term2.append(soil_mass_histogram_2d(env.env.ps, env.env.cfg, BX, BY))
        n = len(term1)
        for i in range(n):
            disp1.append(w2_1d_exact(init1, term1[i], grid1d))
            for j in range(i + 1, n):
                cov1.append(w2_1d_exact(term1[i], term1[j], grid1d))
        t2 = torch.tensor(np.stack(term2), dtype=torch.float32)
        i2 = torch.tensor(np.stack([init2] * n), dtype=torch.float32)
        disp2.extend(sw2_eval(t2, i2, None).tolist())
        for i in range(n):
            for j in range(i + 1, n):
                cov2.append(sw2_eval(t2[i:i + 1], t2[j:j + 1], None).item())
    env.close()
    return {
        "run": run_name, "measure": measure,
        "metric": cfg.metric, "phi_input": getattr(cfg, "phi_input", "hist"),
        "cov_1d": float(np.mean(cov1)), "cov_2d": float(np.mean(cov2)),
        "disp_1d": float(np.mean(disp1)), "disp_2d": float(np.mean(disp2)),
    }


def main():
    args = parse_args()
    runs = BATCH_RUNS if (args.all_batch or args.runs is None) else args.runs

    # Fixed eval geometry, identical for every run.
    tmp_env = SkillDiscoveryEnv(randomize_terrain=True)
    grid2d = torch.tensor(bin_centers_2d(tmp_env.env.cfg, BX, BY),
                          dtype=torch.float32)
    tmp_env.close()
    sw2_eval = SlicedW2(grid2d, n_projections=128, seed=0)
    rng = np.random.default_rng(SKILL_SEED)
    v = rng.normal(size=(N_SKILLS, 4))
    dirs = (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)

    rows = []
    for r in runs:
        if not (ROOT / "runs" / r / args.ckpt).exists():
            print(f"{r:28s} SKIP (no checkpoint)")
            continue
        row = eval_run(r, args.ckpt, sw2_eval, dirs)
        rows.append(row)
        print(f"{row['run']:28s} [{row['measure']}/{row['metric']}/"
              f"{row['phi_input']}] cov1d {row['cov_1d']:.3f}  "
              f"cov2d {row['cov_2d']:.3f}  disp1d {row['disp_1d']:.3f}  "
              f"disp2d {row['disp_2d']:.3f}")

    out = ROOT / "runs" / "compare_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
