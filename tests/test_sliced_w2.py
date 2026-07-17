"""Unit tests for tmsd.metrics.SlicedW2 (2D measure ground metric)."""

import numpy as np
import pytest
import torch

from tmsd.metrics import SlicedW2

BX, BY = 16, 8
K = BX * BY


def _grid2d():
    cx = np.arange(BX) + 0.5
    cy = np.arange(BY) + 0.5
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    return torch.tensor(np.stack([gx.ravel(), gy.ravel()], -1),
                        dtype=torch.float32)


GRID = _grid2d()
SW2 = SlicedW2(GRID, n_projections=256, seed=0)


def _delta(ix, iy):
    h = np.zeros(K, dtype=np.float32)
    h[ix * BY + iy] = 1.0
    return h


def _t(*hists):
    return torch.tensor(np.stack(hists), dtype=torch.float32)


def test_identical_is_zero():
    rng = np.random.default_rng(0)
    p = rng.random(K).astype(np.float32)
    p /= p.sum()
    assert SW2(_t(p), _t(p), GRID).item() == pytest.approx(0.0, abs=1e-4)


def test_shifted_deltas_scale():
    # Two deltas separated by v: SW2 = |v| * sqrt(E[cos^2 theta]) ~ |v|/sqrt(2).
    p, q = _delta(2, 3), _delta(10, 3)  # |v| = 8 along x
    d = SW2(_t(p), _t(q), GRID).item()
    assert d == pytest.approx(8.0 / np.sqrt(2.0), rel=0.12)


def test_monotone_in_shift():
    p = _delta(2, 3)
    d_small = SW2(_t(p), _t(_delta(5, 3)), GRID).item()
    d_large = SW2(_t(p), _t(_delta(12, 3)), GRID).item()
    assert d_large > d_small * 1.5


def test_sees_vertical_structure():
    # Same x-marginal, different y: 1D x-metric is blind to this, SW2 not.
    p, q = _delta(8, 1), _delta(8, 6)
    assert SW2(_t(p), _t(q), GRID).item() > 1.0


def test_symmetry():
    rng = np.random.default_rng(1)
    p = rng.random(K).astype(np.float32); p /= p.sum()
    q = rng.random(K).astype(np.float32); q /= q.sum()
    assert SW2(_t(p), _t(q), GRID).item() == pytest.approx(
        SW2(_t(q), _t(p), GRID).item(), rel=1e-5)


def test_empty_is_zero():
    z = np.zeros(K, dtype=np.float32)
    p = _delta(4, 4)
    assert SW2(_t(z), _t(p), GRID).item() == 0.0


def test_batching_matches_loop():
    rng = np.random.default_rng(2)
    ps, qs = [], []
    for _ in range(4):
        a = rng.random(K).astype(np.float32); a /= a.sum()
        b = rng.random(K).astype(np.float32); b /= b.sum()
        ps.append(a); qs.append(b)
    batch = SW2(_t(*ps), _t(*qs), GRID)
    for i in range(4):
        single = SW2(_t(ps[i]), _t(qs[i]), GRID).item()
        assert batch[i].item() == pytest.approx(single, rel=1e-5)


def test_deterministic_across_instances():
    a, b = _delta(3, 2), _delta(9, 5)
    other = SlicedW2(GRID, n_projections=256, seed=0)
    assert SW2(_t(a), _t(b), GRID).item() == pytest.approx(
        other(_t(a), _t(b), GRID).item(), rel=1e-6)
