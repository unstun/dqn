from __future__ import annotations

import math

import numpy as np
import torch

from forest_vehicle_dqn.cli.train import _forest_policy_action_from_q


class _DummyModel:
    dt = 0.2
    wheelbase_m = 2.8
    v_max_m_s = 2.0
    delta_max_rad = math.radians(27.0)
    delta_dot_max_rad_s = math.radians(60.0)


class _DummyForestEnv:
    def __init__(self) -> None:
        self.model = _DummyModel()
        self.action_table = np.asarray(
            [
                [0.0, 0.0],  # a0: inadmissible argmax
                [0.0, 0.0],  # smooth candidate
                [float(self.model.delta_dot_max_rad_s), 0.0],  # aggressive candidate
            ],
            dtype=np.float32,
        )
        self._x_m = 0.0
        self._y_m = 0.0
        self._psi_rad = 0.0
        self._v_m_s = 1.0
        self._delta_rad = 0.0

    def _goal_pose_reached(self, *, d_goal_m: float, alpha_rad: float) -> bool:
        return False

    def _distance_to_goal_m(self) -> float:
        return 10.0

    def _goal_relative_angle_rad(self) -> float:
        return 0.0

    def _goal_stop_reached(self, *, v_m_s: float, delta_rad: float) -> bool:
        return False

    def is_action_admissible(
        self,
        action_id: int,
        *,
        horizon_steps: int,
        min_od_m: float,
        min_progress_m: float,
    ) -> bool:
        return int(action_id) != 0

    def admissible_action_mask(
        self,
        *,
        horizon_steps: int,
        min_od_m: float,
        min_progress_m: float,
        fallback_to_safe: bool,
    ) -> np.ndarray:
        return np.asarray([False, True, True], dtype=np.bool_)

    def _fallback_action_short_rollout(self, *, horizon_steps: int, min_od_m: float = 0.0) -> int:
        return 1

    def _od_and_collision_at_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> tuple[float, bool]:
        return 1.0, False


class _DummyForestEnvProgressEmpty:
    def __init__(self) -> None:
        self.model = _DummyModel()
        self.action_table = np.asarray(
            [
                [0.0, 0.0],  # a0: inadmissible argmax
                [0.0, 0.0],  # safe fallback candidate (lower Q)
                [0.0, 0.0],  # safe fallback candidate (higher Q)
            ],
            dtype=np.float32,
        )
        self._x_m = 0.0
        self._y_m = 0.0
        self._psi_rad = 0.0
        self._v_m_s = 1.0
        self._delta_rad = 0.0

    def _goal_pose_reached(self, *, d_goal_m: float, alpha_rad: float) -> bool:
        return False

    def _distance_to_goal_m(self) -> float:
        return 10.0

    def _goal_relative_angle_rad(self) -> float:
        return 0.0

    def _goal_stop_reached(self, *, v_m_s: float, delta_rad: float) -> bool:
        return False

    def is_action_admissible(
        self,
        action_id: int,
        *,
        horizon_steps: int,
        min_od_m: float,
        min_progress_m: float,
    ) -> bool:
        return False

    def admissible_action_mask(
        self,
        *,
        horizon_steps: int,
        min_od_m: float,
        min_progress_m: float,
        fallback_to_safe: bool,
    ) -> np.ndarray:
        if bool(fallback_to_safe):
            return np.asarray([False, True, True], dtype=np.bool_)
        return np.asarray([False, False, False], dtype=np.bool_)

    def _fallback_action_short_rollout(self, *, horizon_steps: int, min_od_m: float = 0.0) -> int:
        return 1

    def _od_and_collision_at_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> tuple[float, bool]:
        return 1.0, False


class _DummyForestEnvAllMasksEmpty:
    def __init__(self) -> None:
        self.model = _DummyModel()
        self.action_table = np.asarray(
            [
                [0.0, 0.0],  # a0: inadmissible argmax
                [0.0, 0.0],  # last-resort fallback action
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self._x_m = 0.0
        self._y_m = 0.0
        self._psi_rad = 0.0
        self._v_m_s = 1.0
        self._delta_rad = 0.0

    def _goal_pose_reached(self, *, d_goal_m: float, alpha_rad: float) -> bool:
        return False

    def _distance_to_goal_m(self) -> float:
        return 10.0

    def _goal_relative_angle_rad(self) -> float:
        return 0.0

    def _goal_stop_reached(self, *, v_m_s: float, delta_rad: float) -> bool:
        return False

    def is_action_admissible(
        self,
        action_id: int,
        *,
        horizon_steps: int,
        min_od_m: float,
        min_progress_m: float,
    ) -> bool:
        return False

    def admissible_action_mask(
        self,
        *,
        horizon_steps: int,
        min_od_m: float,
        min_progress_m: float,
        fallback_to_safe: bool,
    ) -> np.ndarray:
        return np.asarray([False, False, False], dtype=np.bool_)

    def _fallback_action_short_rollout(self, *, horizon_steps: int, min_od_m: float = 0.0) -> int:
        return 1

    def _od_and_collision_at_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> tuple[float, bool]:
        return 1.0, False


def test_turn_aware_topk_disabled_keeps_highest_q_admissible() -> None:
    env = _DummyForestEnv()
    q = torch.tensor([3.0, 1.95, 2.0], dtype=torch.float32)
    action, argmax_inadmissible = _forest_policy_action_from_q(
        env,
        q,
        forest_adm_horizon=10,
        forest_topk=3,
        forest_topk_turn_penalty=0.0,
        forest_min_od_m=0.0,
        forest_min_progress_m=0.0,
        forest_no_fallback=False,
    )
    assert int(action) == 2
    assert bool(argmax_inadmissible) is True


def test_turn_aware_topk_enabled_prefers_smoother_candidate() -> None:
    env = _DummyForestEnv()
    q = torch.tensor([3.0, 1.95, 2.0], dtype=torch.float32)
    action, argmax_inadmissible = _forest_policy_action_from_q(
        env,
        q,
        forest_adm_horizon=10,
        forest_topk=3,
        forest_topk_turn_penalty=1.0,
        forest_min_od_m=0.0,
        forest_min_progress_m=0.0,
        forest_no_fallback=False,
    )
    assert int(action) == 1
    assert bool(argmax_inadmissible) is True


def test_turn_aware_topk_does_not_apply_in_strict_mode() -> None:
    env = _DummyForestEnv()
    q = torch.tensor([3.0, 1.95, 2.0], dtype=torch.float32)
    action, argmax_inadmissible = _forest_policy_action_from_q(
        env,
        q,
        forest_adm_horizon=10,
        forest_topk=3,
        forest_topk_turn_penalty=1.0,
        forest_min_od_m=0.0,
        forest_min_progress_m=0.0,
        forest_no_fallback=True,
    )
    assert int(action) == 0
    assert bool(argmax_inadmissible) is True


def test_fallback_to_safe_mask_avoids_inadmissible_argmax() -> None:
    env = _DummyForestEnvProgressEmpty()
    q = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32)
    action, argmax_inadmissible = _forest_policy_action_from_q(
        env,
        q,
        forest_adm_horizon=10,
        forest_topk=3,
        forest_topk_turn_penalty=0.0,
        forest_min_od_m=0.0,
        forest_min_progress_m=0.0,
        forest_no_fallback=False,
    )
    assert int(action) == 2
    assert bool(argmax_inadmissible) is True


def test_fallback_short_rollout_used_when_all_masks_empty() -> None:
    env = _DummyForestEnvAllMasksEmpty()
    q = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    action, argmax_inadmissible = _forest_policy_action_from_q(
        env,
        q,
        forest_adm_horizon=10,
        forest_topk=3,
        forest_topk_turn_penalty=0.0,
        forest_min_od_m=0.0,
        forest_min_progress_m=0.0,
        forest_no_fallback=False,
    )
    assert int(action) == 1
    assert bool(argmax_inadmissible) is True
