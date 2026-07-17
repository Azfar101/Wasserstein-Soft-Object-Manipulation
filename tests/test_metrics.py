"""Unit tests for tmsd.metrics — W₂ correctness against exact references."""

import numpy as np
import pytest
import torch
from scipy import stats

from tmsd.metrics import w2_1d, w2_1d_exact, euclidean, temporal

K = 64
GRID_NP = np.linspace(0.5, K - 0.5, K)  # unit-width bins, centers
GRID = torch.tensor(GRID_NP, dtype=torch.float32)


def _rand_hist(rng, k=K):
    h = rng.random(k)
    return h / h.sum()


def _t(*hists):
    return torch.tensor(np.stack(hists), dtype=torch.float32)


def test_identical_is_zero():
    rng = np.random.default_rng(0)
    p = _rand_hist(rng)
    d = w2_1d(_t(p), _t(p), GRID)
    assert d.item() == pytest.approx(0.0, abs=1e-4)


def test_shifted_deltas_give_shift_distance():
    p = np.zeros(K); p[10] = 1.0
    q = np.zeros(K); q[30] = 1.0
    d = w2_1d(_t(p), _t(q), GRID)
    assert d.item() == pytest.approx(20.0, rel=1e-3)


def test_symmetry():
    rng = np.random.default_rng(1)
    p, q = _rand_hist(rng), _rand_hist(rng)
    d_pq = w2_1d(_t(p), _t(q), GRID).item()
    d_qp = w2_1d(_t(q), _t(p), GRID).item()
    assert d_pq == pytest.approx(d_qp, rel=1e-5)


def test_matches_exact_reference():
    rng = np.random.default_rng(2)
    for _ in range(20):
        p, q = _rand_hist(rng), _rand_hist(rng)
        approx = w2_1d(_t(p), _t(q), GRID, n_quantiles=4096).item()
        exact = w2_1d_exact(p, q, GRID_NP)
        assert approx == pytest.approx(exact, rel=5e-3, abs=1e-3)


def test_w1_consistency_with_scipy():
    # scipy computes W1; our W2 must upper-bound W1 (Jensen) and agree
    # exactly for two deltas (all mass moves the same distance).
    p = np.zeros(K); p[5] = 1.0
    q = np.zeros(K); q[50] = 1.0
    w1 = stats.wasserstein_distance(GRID_NP, GRID_NP, p, q)
    w2 = w2_1d(_t(p), _t(q), GRID).item()
    assert w2 == pytest.approx(w1, rel=1e-3)

    rng = np.random.default_rng(3)
    for _ in range(10):
        a, b = _rand_hist(rng), _rand_hist(rng)
        w1 = stats.wasserstein_distance(GRID_NP, GRID_NP, a, b)
        w2 = w2_1d(_t(a), _t(b), GRID, n_quantiles=4096).item()
        assert w2 >= w1 - 1e-4


def test_empty_histograms_are_distance_zero():
    z = np.zeros(K)
    p = _rand_hist(np.random.default_rng(4))
    assert w2_1d(_t(z), _t(p), GRID).item() == 0.0
    assert w2_1d(_t(z), _t(z), GRID).item() == 0.0


def test_triangle_inequality_sampled():
    rng = np.random.default_rng(5)
    for _ in range(10):
        a, b, c = _rand_hist(rng), _rand_hist(rng), _rand_hist(rng)
        dab = w2_1d_exact(a, b, GRID_NP)
        dbc = w2_1d_exact(b, c, GRID_NP)
        dac = w2_1d_exact(a, c, GRID_NP)
        assert dac <= dab + dbc + 1e-9


def test_batching_matches_loop():
    rng = np.random.default_rng(6)
    ps = [_rand_hist(rng) for _ in range(5)]
    qs = [_rand_hist(rng) for _ in range(5)]
    batch = w2_1d(_t(*ps), _t(*qs), GRID)
    for i in range(5):
        single = w2_1d(_t(ps[i]), _t(qs[i]), GRID).item()
        assert batch[i].item() == pytest.approx(single, rel=1e-5)


def test_ablation_metrics_shapes():
    rng = np.random.default_rng(7)
    p, q = _t(_rand_hist(rng), _rand_hist(rng)), _t(_rand_hist(rng), _rand_hist(rng))
    assert euclidean(p, q, GRID).shape == (2,)
    assert torch.all(temporal(p, q, GRID) == 1.0)
