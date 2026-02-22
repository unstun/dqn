from __future__ import annotations

import numpy as np

from forest_vehicle_dqn.cli.train import _compute_adaptive_no_progress_penalty


def test_adaptive_penalty_disabled_returns_base() -> None:
    penalty = _compute_adaptive_no_progress_penalty(
        base_penalty=0.40,
        enabled=False,
        dist_ratio=0.95,
        dist_gain=0.20,
        min_penalty=0.35,
        max_penalty=0.60,
    )
    assert np.isclose(float(penalty), 0.40, atol=1e-8)


def test_adaptive_penalty_enabled_increases_with_distance_ratio() -> None:
    short_penalty = _compute_adaptive_no_progress_penalty(
        base_penalty=0.35,
        enabled=True,
        dist_ratio=0.20,
        dist_gain=0.12,
        min_penalty=0.30,
        max_penalty=0.60,
    )
    long_penalty = _compute_adaptive_no_progress_penalty(
        base_penalty=0.35,
        enabled=True,
        dist_ratio=0.90,
        dist_gain=0.12,
        min_penalty=0.30,
        max_penalty=0.60,
    )
    assert float(long_penalty) > float(short_penalty)
    assert float(short_penalty) >= 0.35


def test_adaptive_penalty_invalid_inputs_fallback_and_clip() -> None:
    fallback_penalty = _compute_adaptive_no_progress_penalty(
        base_penalty=0.40,
        enabled=True,
        dist_ratio=float("nan"),
        dist_gain=0.30,
        min_penalty=-1.0,
        max_penalty=-1.0,
    )
    clipped_penalty = _compute_adaptive_no_progress_penalty(
        base_penalty=0.40,
        enabled=True,
        dist_ratio=1.00,
        dist_gain=0.50,
        min_penalty=0.20,
        max_penalty=0.55,
    )
    assert np.isclose(float(fallback_penalty), 0.40, atol=1e-8)
    assert np.isclose(float(clipped_penalty), 0.55, atol=1e-8)
