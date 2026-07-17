# Findings: TMSD Experimental Campaign (2026-07-09 → 2026-07-11)

## THE RESULT (updated 07-11): the 2×2 is complete

Skill coverage (physical 1D W₂, frozen protocol, 3 seeds per cell):

|                          | soil-only φ      | full-state φ |
|--------------------------|------------------|--------------|
| **distance-maximizing**  | **0.29–0.35 ✓**  | 0.040 ✗      |
| **MI (continuous DIAYN)**| 0.020 ✗          | 0.030 ✗      |

**Neither ingredient suffices alone; both are necessary; within the working
cell the choice of ground metric (W₂/Euclidean/temporal) is irrelevant.**
MI-soil mechanism: discriminator reaches 0.98 cosine (near-perfect skill ID)
from *microscopic* soil differences while displacement stays at ambient
(~0.04 m) — MI saturates at discriminability, never demands macroscopic
separation. The LSD/METRA critique of MI, demonstrated on a material field.
This partially vindicates the METRA lineage (distance-maximization is
load-bearing) while refuting its metric emphasis (which distance is not).

Caveat: our MI cell is DIAYN-style skill-MI q(z|state). MUSIC's
agent–surrounding MI is a different objective; a MUSIC-faithful variant
remains future work.

Status of every claim from `Ideation.md` after 15 training runs (3 seeds where it
matters), a frozen two-yardstick evaluation protocol, and a zero-shot goal-reaching
study. Numbers below are from `runs/compare_summary.csv` (protocol:
`scripts/compare_runs.py`, fixed before results were seen) and training telemetry.

## Headline verdicts

| # | Claim (Ideation.md) | Verdict | Evidence |
|---|---|---|---|
| 1 | Physical W₂ ground metric → more diverse material skills | **Dead** | Coverage tie across 3 seeds: W₂ 0.326±0.036, Euclid 0.352±0.034, temporal 0.291±0.094 m |
| 2 | W₂ metric → better zero-shot steering | **Dead** | Goal-reaching tie (57.8 / 63.2 / 63.4 % improvement); all φ's ≈0.99 correlated with physical W₂ regardless of training metric |
| 3 | W₂ revives on a richer (2D) measure | **Dead — actively harmful** | Sliced-W₂ coverage 0.044 on its own 2D ruler vs Euclidean-2D 0.245 (5× worse) |
| 4 | External-state-only discriminator matters | **Confirmed, ~8×** | Full-state φ: 0.040±0.046 coverage, 2/3 seeds ≈ zero soil contact; soil-only φ never failed in 9/9 runs |
| 5 | Skill sequencing reaches goals (demonstration layer) | **Confirmed, 3 seeds** | Zero-shot z = φ(goal)−φ(now): 39–63 % W₂-to-goal reduction across 9 checkpoints, no goal training |
| 6 | Contraction certificate (monotone W₂ decrease) | **Confirmed empirically, 3 seeds** | 79–94 % of control intervals monotone, all methods, all seeds; φ-isometry (≈0.99) replicates everywhere |

## Why the metric doesn't matter here (the explanatory finding)

Distortion analysis over visited states: every trained φ — including Euclidean- and
temporal-constrained ones — ends ≈0.99 Pearson/Spearman-correlated with physical W₂.
The reachable soil manifold under a 200-step arm episode is effectively 2–3
dimensional; on such a low-dimensional submanifold all sensible metrics are monotone
reparametrizations of each other, so any Lipschitz diversity objective recovers
transport geometry without being asked. Metric choice can only matter when policies
visit regions where metrics *disagree* — which these never do.

## Why sliced-W₂ actively failed in 2D (hypothesis revised by telemetry)

Initial hypothesis — "avalanche farming" (trigger gravitational relaxation for cheap
mass-flux) — is **refuted** by telemetry: the sliced-W₂ run's displacement (0.041
m/episode) is near ambient settling, not high-flux.

Supported mechanism — **reward-signal starvation / optimization failure**: the run
extracted only ~7 % of its constraint budget as directed reward (0.0072 per step
against a 100·d cap of ~0.107), versus ~45 % for Euclidean-2D. Under the SW₂
constraint, φ failed to align its displacement with the skill vector — rewards
stayed ~18× below the 1D-W₂ runs and the policy never acquired a digging gradient.
The physical transport metric did not mis-reward good behavior; it starved learning.

## Reward-per-physical-work table (mechanism fingerprints)

Episode reward per meter of net soil transport (last quarter of training):

| config | reward / m soil | reading |
|---|---|---|
| soil-φ + W₂ (1D) | 130–210 | reward ∝ physical work — **physically calibrated** |
| soil-φ + Euclidean | 30–44 | proportional, different units |
| soil-φ + temporal | 10k–16k | reward *gated* by soil change, magnitude decoupled |
| full-state φ | 280k–340k | reward decoupled from soil entirely (arm harvesting) |

One real W₂-metric property survives: with the W₂ constraint, rewards (hence
Q-values) carry physical units — meters of mass transport. Calibration, not
performance. Candidate paragraph for the certificate/geometry section.

## What the paper is now

**Headline:** *State factorization, not metric choice, drives unsupervised skill
discovery on granular media — with certified zero-shot shaping.*

1. First reward-free skill discovery on granular/deformable media (gap survives).
2. Factorization ablation: ~8× coverage effect, 12 runs, error bars.
3. Metric-agnosticism finding + low-dim-manifold explanation (+2D harm result) —
   empirical counterpoint to the METRA-lineage emphasis; aligned with "Can a MISL
   Fly?" from physical evidence.
4. Zero-shot goal-reaching + ~90 % monotone contraction (certificate layer).

## Remaining work

- [x] DIAYN-style MI baselines — done as MI 2×2 (3 seeds/cell, `runs/mi_*`):
      both MI cells fail (see THE RESULT above)
- [x] Demo figure — `runs/tmsd_w2_rt_d4/demo_shaping.png`: zero-shot trench +51 %,
      berm +30 %, flatten no-op (below skill granularity); contraction curves
      plateau at ~0.23 m ≈ empirical r(ε) — matches certificate-theory shape
- [ ] Second domain: 2D dough (generality, mandatory per Ideation §5.5)
- [ ] Demo figure: user-drawn target profiles → zero-shot sculpting + contraction plot
- [ ] Contraction theorem, formal write-up
- [ ] Multi-seed goal-reaching eval; 2D-measure goal eval
- [ ] Investigate SW₂ optimization failure (why φ can't exploit the constraint budget)
      — honest open question, one paragraph
- [ ] Pre-submission arXiv scan (Ideation caveat #1)

## Run inventory

`runs/`: tmsd_w2_s0 (500k pilot, fixed terrain, d2); {tmsd_w2, abl_euclid,
abl_temporal}_rt_d4[_s1,_s2] (1D trio × 3 seeds); abl_fullstate_temporal[_s1,_s2]
(factorization baseline × 3 seeds); {tmsd_sw2, abl_euclid, abl_temporal}_2d_s0
(2D-measure trio); goal_eval/ (steering + distortion); compare_summary.csv
(frozen-protocol scores, both yardsticks).
