from __future__ import annotations

import numpy as np

from forest_vehicle_dqn.env import AMRBicycleEnv
from forest_vehicle_dqn.maps import ArrayGridMapSpec


def _border_walls_grid(size: int = 64) -> np.ndarray:
    n = max(5, int(size))
    grid = np.zeros((n, n), dtype=np.uint8)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return grid


def test_admissible_action_mask_fallback_relaxes_min_od_when_needed() -> None:
    grid = _border_walls_grid(64)
    spec = ArrayGridMapSpec(
        name="unit_border_walls",
        grid_y0_bottom=grid,
        start_xy=(32, 32),
        goal_xy=(48, 48),
    )
    env = AMRBicycleEnv(spec, max_steps=200)
    env.reset(seed=0)

    # Make the "safe" filter empty even when collision-free actions exist.
    min_od_m = 1e6

    mask_no_fallback = env.admissible_action_mask(
        horizon_steps=1,
        min_od_m=min_od_m,
        min_progress_m=0.0,
        fallback_to_safe=False,
        allow_reverse=False,
    )
    assert bool(mask_no_fallback.any()) is False

    mask_fallback = env.admissible_action_mask(
        horizon_steps=1,
        min_od_m=min_od_m,
        min_progress_m=0.0,
        fallback_to_safe=True,
        allow_reverse=False,
    )
    assert bool(mask_fallback.any()) is True
