"""
Ground metrics d(g, g′) between material states.

The TMSD Lipschitz constraint is ‖φ(g) − φ(g′)‖ ≤ d(g, g′); everything
here computes d for batches of mass histograms on a fixed uniform grid.
The metric is deliberately a plug (see ``GROUND_METRICS``) because the
paper's load-bearing ablation is swapping the physical transport metric
for Euclidean/temporal alternatives.

For 1D distributions W₂ is closed-form via quantile functions:
W₂²(μ, ν) = ∫₀¹ (Q_μ(u) − Q_ν(u))² du. ``w2_1d`` evaluates the integral
by sampling the inverse CDFs at midpoint quantiles (exact in the limit;
with n_quantiles ≳ 8× bins the error is far below a bin width).
``w2_1d_exact`` does the exact piecewise-constant integral for tests.

No gradients flow through d: it is a fixed physical cost, and only φ is
trained against it.
"""

from __future__ import annotations

import numpy as np
import torch


# ── W₂ on 1D histograms (torch, batched, no-grad) ───────────────────
@torch.no_grad()
def w2_1d(p: torch.Tensor, q: torch.Tensor, grid: torch.Tensor,
          n_quantiles: int = 512) -> torch.Tensor:
    """Batched 2-Wasserstein distance between 1D mass histograms.

    Parameters
    ----------
    p, q : (B, K) nonnegative, each row summing to 1 (rows summing to 0
        are treated as equal to anything summing to 0, distance 0).
    grid : (K,) world coordinates of the bin centers (meters).
    n_quantiles : quadrature resolution for the quantile integral.

    Returns
    -------
    (B,) tensor of W₂ distances in world units (meters).
    """
    cdf_p = torch.cumsum(p, dim=-1).clamp(max=1.0)
    cdf_q = torch.cumsum(q, dim=-1).clamp(max=1.0)
    u = (torch.arange(n_quantiles, device=p.device, dtype=p.dtype) + 0.5) / n_quantiles
    u = u.expand(p.shape[0], -1).contiguous()
    idx_p = torch.searchsorted(cdf_p.contiguous(), u).clamp(max=p.shape[-1] - 1)
    idx_q = torch.searchsorted(cdf_q.contiguous(), u).clamp(max=q.shape[-1] - 1)
    qf_p = grid[idx_p]
    qf_q = grid[idx_q]
    w2_sq = ((qf_p - qf_q) ** 2).mean(dim=-1)
    # Degenerate rows (no mass) carry no transport cost.
    empty = (p.sum(-1) <= 0) | (q.sum(-1) <= 0)
    return torch.where(empty, torch.zeros_like(w2_sq), w2_sq).sqrt()


def w2_1d_exact(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    """Exact W₂ between two atomic 1D distributions on a common grid.

    Integrates (Q_p − Q_q)² over the union of both CDFs' breakpoints —
    the quantile functions are piecewise constant, so this is exact.
    Reference implementation for tests; not batched.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    # Breakpoints of the merged quantile partition.
    us = np.unique(np.concatenate([[0.0], cdf_p, cdf_q, [1.0]]))
    us = us[(us > 0.0) & (us <= 1.0)]
    last = len(grid) - 1
    lo = 0.0
    total = 0.0
    for u in us:
        mid = 0.5 * (lo + u)
        xi = grid[min(np.searchsorted(cdf_p, mid), last)]
        xj = grid[min(np.searchsorted(cdf_q, mid), last)]
        total += (xi - xj) ** 2 * (u - lo)
        lo = u
    return float(np.sqrt(total))


# ── Euclidean ablation metric ────────────────────────────────────────
@torch.no_grad()
def euclidean(p: torch.Tensor, q: torch.Tensor, grid: torch.Tensor,
              **_) -> torch.Tensor:
    """L2 distance between raw histograms (the 'metric is not doing the
    work' ablation). Ignores the grid — no notion of transport cost."""
    return torch.linalg.vector_norm(p - q, dim=-1)


# ── Temporal ablation metric (METRA's original: adjacent steps ≤ 1) ──
@torch.no_grad()
def temporal(p: torch.Tensor, q: torch.Tensor, grid: torch.Tensor,
             **_) -> torch.Tensor:
    """Constant distance 1 between adjacent states — recovers vanilla
    METRA's temporal-distance constraint for trajectory-adjacent pairs."""
    return torch.ones(p.shape[0], device=p.device, dtype=p.dtype)


class SlicedW2:
    """Sliced 2-Wasserstein between mass histograms on a fixed 2D grid.

    SW₂²(μ, ν) = E_u[ W₂²(u#μ, u#ν) ] over random unit directions u —
    each projection is a 1D transport problem solved in closed form.
    Projection directions and per-direction sort orders of the (static)
    grid are precomputed once, so a batch evaluation is gather + cumsum
    + searchsorted. Note SW₂ underestimates true W₂ by a dimension
    factor (√2 for two deltas in 2D); a constant scale, harmless as a
    ground metric.

    Callable with the same signature as the other ground metrics.
    """

    def __init__(self, grid2d: torch.Tensor, n_projections: int = 64,
                 seed: int = 0):
        assert grid2d.ndim == 2 and grid2d.shape[1] == 2, "need (K, 2) grid"
        g = torch.Generator(device="cpu").manual_seed(seed)
        ang = torch.rand(n_projections, generator=g) * torch.pi  # dirs mod π
        u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        proj = (grid2d.cpu().double() @ u.double().T).T          # (P, K)
        order = torch.argsort(proj, dim=-1)                      # (P, K)
        self.order = order.to(grid2d.device)
        self.sorted_coords = torch.gather(proj, -1, order).to(
            grid2d.device, dtype=torch.float32)                  # (P, K)

    @torch.no_grad()
    def __call__(self, p: torch.Tensor, q: torch.Tensor, grid: torch.Tensor,
                 n_quantiles: int = 512, **_) -> torch.Tensor:
        B, K = p.shape
        P = self.order.shape[0]
        idx = self.order.unsqueeze(0).expand(B, -1, -1)          # (B, P, K)
        cdf_p = p.unsqueeze(1).expand(-1, P, -1).gather(-1, idx).cumsum(-1)
        cdf_q = q.unsqueeze(1).expand(-1, P, -1).gather(-1, idx).cumsum(-1)
        cdf_p = cdf_p.clamp(max=1.0).contiguous()
        cdf_q = cdf_q.clamp(max=1.0).contiguous()
        u = (torch.arange(n_quantiles, device=p.device, dtype=p.dtype) + 0.5)
        u = (u / n_quantiles).expand(B, P, -1).contiguous()
        ip = torch.searchsorted(cdf_p, u).clamp(max=K - 1)
        iq = torch.searchsorted(cdf_q, u).clamp(max=K - 1)
        coords = self.sorted_coords.unsqueeze(0).expand(B, -1, -1)
        qf_p = coords.gather(-1, ip)
        qf_q = coords.gather(-1, iq)
        sw2_sq = ((qf_p - qf_q) ** 2).mean(dim=(-1, -2))
        empty = (p.sum(-1) <= 0) | (q.sum(-1) <= 0)
        return torch.where(empty, torch.zeros_like(sw2_sq), sw2_sq).sqrt()


GROUND_METRICS = {
    "w2": w2_1d,
    "euclidean": euclidean,
    "temporal": temporal,
    "sliced_w2": SlicedW2,  # class: instantiated with the 2D grid by the trainer
}
