from __future__ import annotations

import math

import numpy as np
import torch

from forest_vehicle_dqn.cli.infer import _forest_choose_replacement_candidate as infer_choose
from forest_vehicle_dqn.cli.train import _forest_choose_replacement_candidate as train_choose


class _DummyModel:
    dt = 1.0
    wheelbase_m = 1.0
    v_max_m_s = 10.0
    delta_max_rad = math.radians(27.0)
    delta_dot_max_rad_s = math.radians(45.0)


class _DummyEnvProgressQ:
    """Stub env for replacement selection (progress_q + progress_q_clearance).

    With dt=1, v0=0, psi0=0, bicycle_integrate_one_step yields x_next == accel.
    """

    def __init__(self) -> None:
        self.model = _DummyModel()
        self.action_table = np.asarray(
            [
                [0.0, 1.0],  # a0: best progress, high clearance
                [0.0, 2.0],  # a1: best progress, low clearance
                [0.0, 3.0],  # a2: worse progress
            ],
            dtype=np.float32,
        )
        self._x_m = 0.0
        self._y_m = 0.0
        self._psi_rad = 0.0
        self._v_m_s = 0.0
        self._delta_rad = 0.0
        self._progress_by_x = {
            1: 0.0,
            2: 0.0,
            3: 1.0,
        }
        self._od_by_x = {
            1: 2.0,
            2: 0.5,
            3: 1.0,
        }

    def _od_and_collision_at_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> tuple[float, bool]:
        xk = int(round(float(x_m)))
        return float(self._od_by_x.get(xk, 1.0)), False

    def _progress_dist_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> float:
        xk = int(round(float(x_m)))
        return float(self._progress_by_x.get(xk, float("inf")))


def _assert_progress_q_behavior(choose_fn) -> None:
    env = _DummyEnvProgressQ()
    candidates = [0, 1, 2]

    # progress_q: progress first, then Q (clearance should not override Q).
    q = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float32)
    assert (
        choose_fn(
            env,
            q,
            candidates=list(candidates),
            ranking="progress_q",
            topk_turn_penalty=0.0,
            replace_topq=0,
        )
        == 1
    )

    # progress_q_clearance: progress first, then Q, then clearance.
    q = torch.tensor([2.0, 2.0, 0.0], dtype=torch.float32)
    assert (
        choose_fn(
            env,
            q,
            candidates=list(candidates),
            ranking="progress_q_clearance",
            topk_turn_penalty=0.0,
            replace_topq=0,
        )
        == 0
    )


def test_progress_q_rankings_consistent_in_infer_and_train() -> None:
    _assert_progress_q_behavior(infer_choose)
    _assert_progress_q_behavior(train_choose)

