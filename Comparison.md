# Full Method Comparison (final, 2026-07-13)

Every number from the frozen evaluation protocol (`scripts/compare_runs.py`,
fixed before results were seen): coverage = mean pairwise 1D-W₂ between terminal
soil states, 8 protocol skills × 3 protocol terrains, deterministic policies,
200-step episodes. Raw per-run data: `runs/compare_summary.csv` (both yardsticks)
and `runs/full_comparison.csv` (configs, parameters, costs, telemetry).

## Headline table (mean ± sd over seeds)

| method | lineage | coverage (m) | soil/ep (m) | seeds | train h/run |
|---|---|---|---|---|---|
| **dist-max W₂ + soil φ (TMSD)** | ours | **0.326 ± 0.029** | 0.21 | 3 | 1.5–2.5 |
| **dist-max Euclid + soil φ** | ablation | **0.351 ± 0.030** | 0.21 | 3 | 1.3–2.0 |
| **dist-max temporal + soil φ** | METRA + our fix | **0.290 ± 0.077** | 0.21 | 3 | 1.4–2.0 |
| dist-max full-state (= **vanilla METRA**) | frontier | 0.039 ± 0.036 | 0.05 | 3 | 1.4–1.6 |
| MI + soil φ (DIAYN + our fix) | frontier+fix | 0.021 ± 0.015 | 0.04 | 3 | 1.3–1.8 |
| MI full-state (≈ **vanilla DIAYN**) | frontier | 0.031 ± 0.028 | 0.05 | 3 | 1.3–1.8 |
| dynamics + soil φ (DADS + our fix) | frontier+fix | 0.008 ± 0.008 | 0.04 | 2 | ~1.5 |
| dynamics full-state (≈ **vanilla DADS**) | frontier | 0.130 ± 0.073 ⁽*⁾ | 0.11 | 2 | ~1.5 |
| sliced-W₂ on 2D measure | ours (negative) | 0.064 | 0.04 | 1 | 2.5 |
| dist-max W₂, 500k pilot (fixed terrain, dim 2) | ours | 0.369 | 0.27 | 1 | 3.7 |
| stage-2 composite measure | ours (negative-ish) | 0.065 / 0.315 (unstable) | 0.26 | 2 | 4.9 |
| goal-conditioned SAC from scratch | control exp. | fails to learn | — | 1 | 6.2 |

⁽*⁾ One DADS full-state seed (0.203) moves soil *incidentally* — rich arm-dynamics
skills that happen to contact the bed. Its sibling seed: 0.057. Highest seed
variance in the table; still 2.5× below the working family's mean. Consistent
with the mechanism (reward from proprioceptive dynamics), not a counterexample
to it.

Ambient settling floor (no-op policy): ~0.03–0.08 m depending on terrain.
Both yardsticks (1D W₂ and 2D sliced-W₂) produce identical rankings; cov2d ≈ 0.7·cov1d throughout.

## Configuration parity

All runs: same DEM sim (~1200 grains), same randomized-terrain resets, same
skill_dim=4 (except noted pilot), hidden=256 (~315k parameters total; 594k for
2D-measure φ), lr=3e-4, batch=256, SAC with auto-entropy, 200k env steps =
200k gradient updates, 1000 episodes × 200 steps of training data per run,
one update per env step, RTX 5080 at 30–60 env-steps/s. Total campaign:
33 runs ≈ 70 GPU-hours.

## Secondary results (soil-conditioned distance-max models only)

- **Zero-shot goal-reaching**: 39–63% W₂-to-goal reduction across 21 targets,
  9 checkpoints; 79–94% of control intervals monotone (empirical contraction
  certificate); plateau at r(ε) ≈ 0.2 m (skill granularity).
- **Emergent isometry**: ‖Δφ‖ vs physical W₂ Pearson/Spearman ≈ 0.99 for every
  soil-conditioned model regardless of training metric.
- **Mechanism fingerprints** (episode reward per meter of soil moved):
  W₂ 130–210 (physically calibrated) · Euclid 30–44 · temporal 10–16k (gated,
  uncalibrated) · DADS-full ~1000 · METRA-full 280–340k (fully decoupled).
- **Demo**: zone excavation order → 79% of dig-zone soil removed, dump zone
  within 2.6 points of target; trench command +51%, berm +30%.

## The three findings

1. **Two-ingredient law.** Material-manipulation skills require BOTH a
   cumulative (distance-maximizing) objective AND external-state-only
   conditioning. The 2×3 grid has exactly one working cell (3 metrics × 3
   seeds inside it all work; every other cell fails across seeds).
2. **Metric-agnosticism.** Within the working cell, W₂ vs Euclidean vs temporal
   is a tie; every φ becomes ~isometric to physical transport anyway. On the
   richer 2D measure the transport metric is actively harmful (reward
   starvation). What survives of W₂: physically calibrated value functions.
3. **Slow-state failure taxonomy.** MI *saturates* (identifiable from
   microscopic differences), dynamics-predictability *starves* (slow states
   trivially predictable ⇒ reward ≡ 0), distance-max *telescopes* (reward =
   net terminal displacement ⇒ must accumulate change). Design law: on slow
   external states, integrate change over time and condition only on that state.

## Negative results worth keeping

- Sliced-W₂/2D: constraint-budget extraction 7% vs 45% (Euclid) → starvation.
- Composite (bucket/airborne) measure: unstable across seeds (0.065 vs 0.315);
  did not unlock carry/dump skills at 500k.
- Cold-start goal-conditioned SAC: fails on contact-rich digging (needs
  warm-start from skills — future work with mobile base).
- Skill-MPC with ground-truth lookahead ≈ greedy steering (+6–13% on user-mess
  restore): vocabulary, not selection, is the binding constraint.
  **Diversity ≠ controllability**; gap = r(ε).
