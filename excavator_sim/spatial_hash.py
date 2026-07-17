"""
Uniform-grid spatial hash for broad-phase collision detection.

Particles are binned into a 2D grid of square cells; each particle then only
needs to test its own cell and the neighbouring cells, giving near-O(N)
behaviour. The build routine works on flat SoA arrays so it maps cleanly to
both a Numba CPU kernel and a CUDA kernel.

Grid dimensions are derived from the (compile-time) domain size and cell size
in :mod:`excavator_sim.constants`, so they are module constants the kernels can
close over.
"""

import numpy as np
from numba import njit, prange

from .constants import CELL_SIZE, DOMAIN_WIDTH, DOMAIN_HEIGHT

# Grid dimensions (number of cells along each axis).
GRID_COLS = int(np.ceil(DOMAIN_WIDTH / CELL_SIZE)) + 1
GRID_ROWS = int(np.ceil(DOMAIN_HEIGHT / CELL_SIZE)) + 1
GRID_TOTAL = GRID_COLS * GRID_ROWS

# Maximum particles stored per cell; overflow is silently dropped (rare with a
# well-sized cell, and harmless — it only skips a redundant neighbour test).
MAX_PER_CELL = 64


@njit(cache=True)
def cell_index(x, y):
    """Map a world coordinate to a flat, clamped cell index."""
    col = int(x / CELL_SIZE)
    row = int(y / CELL_SIZE)
    if col < 0:
        col = 0
    elif col >= GRID_COLS:
        col = GRID_COLS - 1
    if row < 0:
        row = 0
    elif row >= GRID_ROWS:
        row = GRID_ROWS - 1
    return row * GRID_COLS + col


@njit(cache=True, parallel=True)
def build_grid(px, py, n, cell_counts, cell_particles):
    """Bin *n* particles into the uniform grid.

    Parameters
    ----------
    px, py : float64[:]
        Particle positions.
    n : int
        Number of live particles.
    cell_counts : int32[GRID_TOTAL]
        Per-cell occupancy (overwritten).
    cell_particles : int32[GRID_TOTAL, MAX_PER_CELL]
        Per-cell particle indices (overwritten up to each cell's count).
    """
    for i in prange(GRID_TOTAL):
        cell_counts[i] = 0

    # Sequential bin: atomic-free on the CPU (the CUDA path uses atomicAdd).
    for i in range(n):
        ci = cell_index(px[i], py[i])
        cnt = cell_counts[ci]
        if cnt < MAX_PER_CELL:
            cell_particles[ci, cnt] = i
            cell_counts[ci] = cnt + 1
