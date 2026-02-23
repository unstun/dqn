from __future__ import annotations

import math

import numpy as np

from forest_vehicle_dqn.env import bilinear_sample_2d_finite_vec, dijkstra8_nocorner_goal_dist_m, grid4_goal_dist_m


def test_grid4_goal_dist_m_detour_with_gap() -> None:
    # 7x7 grid. A wall at y=3 with a single gap at x=6 forces a detour.
    h, w = 7, 7
    trav = np.ones((h, w), dtype=np.bool_)
    trav[3, 0:6] = False

    cell = 0.1
    dist = grid4_goal_dist_m(trav, goal_xy=(0, 6), cell_size_m=cell)

    assert dist.shape == (h, w)
    assert dist.dtype == np.float32
    assert dist[6, 0] == 0.0
    assert math.isclose(float(dist[0, 0]), 18.0 * cell, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(dist[2, 0]), 16.0 * cell, rel_tol=0.0, abs_tol=1e-6)


def test_grid4_goal_dist_m_unreachable_region_is_inf() -> None:
    # Same grid but the wall fully blocks connectivity, making the lower region unreachable.
    h, w = 7, 7
    trav = np.ones((h, w), dtype=np.bool_)
    trav[3, :] = False

    dist = grid4_goal_dist_m(trav, goal_xy=(0, 6), cell_size_m=1.0)

    assert dist[6, 0] == 0.0
    assert math.isinf(float(dist[0, 0]))
    assert math.isinf(float(dist[3, 0]))  # obstacle cell


def test_dijkstra8_nocorner_goal_dist_m_diagonal_step_is_sqrt2() -> None:
    trav = np.ones((2, 2), dtype=np.bool_)
    cost = np.ones((2, 2), dtype=np.float32)
    dist = dijkstra8_nocorner_goal_dist_m(trav, goal_xy=(0, 0), cell_size_m=1.0, cost_factor=cost)
    assert dist.shape == (2, 2)
    assert dist.dtype == np.float32
    assert dist[0, 0] == 0.0
    assert math.isclose(float(dist[1, 1]), math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-6)


def test_dijkstra8_nocorner_goal_dist_m_blocks_corner_cut() -> None:
    trav = np.ones((3, 3), dtype=np.bool_)
    trav[0, 1] = False  # block (1,0)
    trav[1, 0] = False  # block (0,1)
    cost = np.ones((3, 3), dtype=np.float32)
    dist = dijkstra8_nocorner_goal_dist_m(trav, goal_xy=(0, 0), cell_size_m=1.0, cost_factor=cost)
    assert math.isinf(float(dist[1, 1]))


def test_dijkstra8_nocorner_goal_dist_m_accumulates_cost_factor() -> None:
    trav = np.ones((1, 3), dtype=np.bool_)
    cost = np.array([[1.0, 10.0, 1.0]], dtype=np.float32)
    dist = dijkstra8_nocorner_goal_dist_m(trav, goal_xy=(0, 0), cell_size_m=1.0, cost_factor=cost)
    assert dist.shape == (1, 3)
    assert dist.dtype == np.float32
    assert dist[0, 0] == 0.0
    assert math.isclose(float(dist[0, 1]), 10.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(dist[0, 2]), 11.0, rel_tol=0.0, abs_tol=1e-6)


def test_bilinear_sample_2d_finite_vec_replaces_inf_corner_values() -> None:
    arr = np.array([[0.0, float("inf")], [0.0, float("inf")]], dtype=np.float32)
    x = np.array([0.0], dtype=np.float64)
    y = np.array([0.5], dtype=np.float64)
    out = bilinear_sample_2d_finite_vec(arr, x=x, y=y, fill_value=10.0, default=10.0)
    assert out.shape == (1,)
    assert math.isfinite(float(out[0]))
    assert math.isclose(float(out[0]), 0.0, rel_tol=0.0, abs_tol=1e-9)
