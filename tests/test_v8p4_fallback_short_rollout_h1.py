from __future__ import annotations

import math

import numpy as np

from forest_vehicle_dqn.env import AMRBicycleEnv


class _DummyModel:
    dt = 1.0
    wheelbase_m = 1.0
    v_max_m_s = 10.0
    delta_max_rad = math.radians(27.0)


class _DummyEnvLongHorizonAllCollide:
    """A minimal stub for testing `_fallback_action_short_rollout` behavior.

    Setup:
    - h=30 rollout: every constant-action rollout collides -> old impl falls to "clearance-only".
    - h=1 rollout: there exist collision-free actions -> new impl must return one of them.
    """

    def __init__(self) -> None:
        self.model = _DummyModel()
        self.action_table = np.asarray(
            [
                [0.0, 0.0],  # a0: colliding action with (fake) highest clearance
                [0.0, 1.0],  # a1: collision-free but worse progress
                [0.0, 2.0],  # a2: collision-free and best progress
            ],
            dtype=np.float32,
        )
        self._x_m = 0.0
        self._y_m = 0.0
        self._psi_rad = 0.0
        self._v_m_s = 0.0
        self._delta_rad = 0.0
        self._progress_dist = np.zeros((int(self.action_table.shape[0]),), dtype=np.float64)

    def _rollout_constant_actions_end_state(
        self,
        *,
        delta_dot_rad_s: np.ndarray,
        a_m_s2: np.ndarray,
        horizon_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = int(self.action_table.shape[0])
        if int(horizon_steps) == 30:
            self._progress_dist = np.asarray([10.0, 9.0, 8.0], dtype=np.float64)
            x = np.zeros((n,), dtype=np.float64)
            y = np.zeros((n,), dtype=np.float64)
            psi = np.zeros((n,), dtype=np.float64)
            v = np.zeros((n,), dtype=np.float64)
            min_od = np.ones((n,), dtype=np.float64)
            coll = np.ones((n,), dtype=np.bool_)
            reached = np.zeros((n,), dtype=np.bool_)
            return x, y, psi, v, min_od, coll, reached

        if int(horizon_steps) == 1:
            self._progress_dist = np.asarray([10.0, 5.0, 3.0], dtype=np.float64)
            x = np.zeros((n,), dtype=np.float64)
            y = np.zeros((n,), dtype=np.float64)
            psi = np.zeros((n,), dtype=np.float64)
            v = np.zeros((n,), dtype=np.float64)
            min_od = np.asarray([0.0, 0.2, 0.1], dtype=np.float64)
            coll = np.asarray([True, False, False], dtype=np.bool_)
            reached = np.zeros((n,), dtype=np.bool_)
            return x, y, psi, v, min_od, coll, reached

        raise AssertionError(f"unexpected horizon_steps={horizon_steps}")

    def _progress_dist_pose_m_vec(self, x: np.ndarray, y: np.ndarray, psi: np.ndarray) -> np.ndarray:
        return self._progress_dist

    def _od_and_collision_at_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> tuple[float, bool]:
        # In the "clearance-only" branch, `bicycle_integrate_one_step` makes x_next == accel
        # when dt=1, v0=0, and psi=0 (cos=1). Use that to tag the action id.
        if float(x_m) < 0.5:
            return 100.0, True
        if float(x_m) < 1.5:
            return 1.0, False
        return 0.5, False


def test_fallback_short_rollout_prefers_one_step_collision_free_when_long_horizon_all_collide() -> None:
    env = _DummyEnvLongHorizonAllCollide()
    chosen = AMRBicycleEnv._fallback_action_short_rollout(env, horizon_steps=30, min_od_m=1e6)
    assert int(chosen) == 2

