from __future__ import annotations

import math

import numpy as np

from forest_vehicle_dqn.env import grid4_goal_dist_m


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

