"""
Harvest the complete method comparison across every training run:
configuration, network parameter counts, training time/throughput,
training-data volume, training telemetry, and (where present) the
frozen-protocol evaluation results from runs/compare_summary.csv.

Emits runs/full_comparison.csv and prints a compact table.
Rerunnable: run again after new runs/evals to refresh.

Usage:
    python scripts/full_comparison.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# Method labels: run-name prefix -> (family, faithful-to)
FAMILY = [
    ("tmsd_w2_s0", "distance-max (W2)", "ours (pilot, fixed terrain)"),
    ("tmsd_w2_rt", "distance-max (W2)", "ours: TMSD"),
    ("tmsd_sw2_2d", "distance-max (sliced-W2)", "ours: TMSD-2D"),
    ("abl_euclid_2d", "distance-max (Euclidean)", "ablation (2D measure)"),
    ("abl_euclid", "distance-max (Euclidean)", "ablation"),
    ("abl_temporal_2d", "distance-max (temporal)", "METRA-style (2D measure)"),
    ("abl_temporal_rt", "distance-max (temporal)", "METRA + our factorization"),
    ("abl_fullstate", "distance-max (temporal)", "METRA (vanilla, full-state)"),
    ("mi_soil", "MI discriminator", "DIAYN + our factorization"),
    ("mi_full", "MI discriminator", "DIAYN (vanilla, full-state)"),
    ("dads_soil", "skill dynamics", "DADS + our factorization"),
    ("dads_full", "skill dynamics", "DADS (vanilla, full-state)"),
    ("stage2_comp", "distance-max (Euclidean)", "ours: stage-2 composite"),
    ("stage2_1d", "distance-max (Euclidean)", "ours: stage-2 control"),
    ("gc_s0", "goal-conditioned SAC", "task-trained control layer"),
]


def classify(name: str):
    for prefix, fam, faith in FAMILY:
        if name.startswith(prefix):
            return fam, faith
    return "?", "?"


def n_params(sd: dict) -> int:
    return sum(int(np.prod(v.shape)) for v in sd.values())


def harvest_run(d: Path) -> dict | None:
    ckpt_p = d / "ckpt_latest.pt"
    log_p = d / "log.csv"
    ep_p = d / "episodes.csv"
    if not ckpt_p.exists() or not log_p.exists():
        return None
    saved = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    cfg = saved["cfg"]
    is_gc = saved.get("kind") == "gc"

    row = {"run": d.name}
    fam, faith = classify(d.name)
    row["family"] = "goal-conditioned SAC" if is_gc else fam
    row["method"] = faith

    # ── configuration ────────────────────────────────────────────────
    row["objective"] = "gc" if is_gc else getattr(cfg, "objective", "metra")
    row["metric"] = "-" if is_gc else getattr(cfg, "metric", "-")
    row["phi_input"] = "-" if is_gc else getattr(cfg, "phi_input", "hist")
    row["measure"] = "-" if is_gc else getattr(cfg, "measure", "1d")
    row["skill_dim"] = "-" if is_gc else cfg.skill_dim
    row["hidden"] = cfg.hidden
    row["lr"] = cfg.lr
    row["batch"] = cfg.batch_size
    row["gamma"] = cfg.gamma
    row["tau"] = cfg.tau
    row["buffer"] = cfg.buffer_capacity
    row["reward_scale"] = getattr(cfg, "reward_scale", 1.0)
    row["seed"] = cfg.seed

    # ── parameter counts ─────────────────────────────────────────────
    pol = n_params(saved["policy"])
    q = n_params(saved["q"])
    rep = n_params(saved.get("phi", {})) + n_params(saved.get("dyn", {}))
    row["params_policy"] = pol
    row["params_critics"] = q
    row["params_repr"] = rep
    row["params_total"] = pol + q + rep

    # ── training time / data ─────────────────────────────────────────
    log = list(csv.DictReader(open(log_p)))
    if log:
        last = log[-1]
        steps = int(last["step"])
        sps = float(last["sps"])
        row["env_steps"] = steps
        row["steps_per_s"] = sps
        row["train_hours"] = round(steps / sps / 3600, 2)
        row["grad_updates"] = steps  # 1 update per env step post-warmup
    if ep_p.exists():
        eps = list(csv.DictReader(open(ep_p)))
        row["episodes"] = len(eps)
        key = ("terminal_w2_from_start"
               if eps and "terminal_w2_from_start" in eps[0] else None)
        if key:
            tail = [float(e[key]) for e in eps[3 * len(eps) // 4:]]
            row["soil_moved_per_ep_m"] = round(float(np.mean(tail)), 3)
        elif eps and "w2_end" in eps[0]:
            tail = [float(e["improvement"]) for e in eps[3 * len(eps) // 4:]]
            row["gc_improvement"] = round(float(np.mean(tail)), 3)
    # final training reward
    for k in ("sac/reward_mean", "gc/reward_mean"):
        if log and k in log[-1]:
            row["final_reward_per_step"] = round(float(log[-1][k]), 4)
    return row


def main():
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name in ("goal_eval", "smoke"):
            continue
        r = harvest_run(d)
        if r:
            rows.append(r)

    # merge frozen-protocol eval results if present
    cmp_p = RUNS / "compare_summary.csv"
    evals = {}
    if cmp_p.exists():
        for r in csv.DictReader(open(cmp_p)):
            evals[r["run"]] = r
    for row in rows:
        e = evals.get(row["run"])
        if e:
            for k in ("cov_1d", "cov_2d", "disp_1d", "disp_2d"):
                row[f"eval_{k}"] = round(float(e[k]), 3)

    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (k != "run", k != "method", k != "family", k))
    out = RUNS / "full_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{'run':28s} {'family':26s} {'hrs':>5s} {'steps':>7s} "
          f"{'params':>8s} {'soil/ep':>8s} {'cov1d':>6s}")
    for r in rows:
        print(f"{r['run']:28s} {r['family']:26s} "
              f"{r.get('train_hours', 0):5.1f} {r.get('env_steps', 0):7d} "
              f"{r.get('params_total', 0):8d} "
              f"{str(r.get('soil_moved_per_ep_m', r.get('gc_improvement', '-'))):>8s} "
              f"{str(r.get('eval_cov_1d', '-')):>6s}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
