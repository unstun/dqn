from __future__ import annotations

import numpy as np

from forest_vehicle_dqn.cli.train import _resolve_train_suite_no_progress_penalties


def test_suite_penalty_disabled_falls_back_to_base() -> None:
    enabled, short_penalty, long_penalty = _resolve_train_suite_no_progress_penalties(
        base_penalty=0.40,
        enabled=False,
        short_penalty=0.45,
        long_penalty=0.35,
    )
    assert bool(enabled) is False
    assert np.isclose(float(short_penalty), 0.40, atol=1e-8)
    assert np.isclose(float(long_penalty), 0.40, atol=1e-8)


def test_suite_penalty_enabled_uses_overrides() -> None:
    enabled, short_penalty, long_penalty = _resolve_train_suite_no_progress_penalties(
        base_penalty=0.40,
        enabled=True,
        short_penalty=0.45,
        long_penalty=0.35,
    )
    assert bool(enabled) is True
    assert np.isclose(float(short_penalty), 0.45, atol=1e-8)
    assert np.isclose(float(long_penalty), 0.35, atol=1e-8)


def test_suite_penalty_enabled_invalid_override_uses_base() -> None:
    enabled, short_penalty, long_penalty = _resolve_train_suite_no_progress_penalties(
        base_penalty=0.40,
        enabled=True,
        short_penalty=-0.10,
        long_penalty=float("nan"),
    )
    assert bool(enabled) is True
    assert np.isclose(float(short_penalty), 0.40, atol=1e-8)
    assert np.isclose(float(long_penalty), 0.40, atol=1e-8)
