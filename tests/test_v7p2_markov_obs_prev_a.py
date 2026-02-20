from __future__ import annotations

import numpy as np
import unittest

from forest_vehicle_dqn.env import AMRBicycleEnv
from forest_vehicle_dqn.maps import get_map_spec
from forest_vehicle_dqn.networks import infer_flat_obs_cnn_layout


def _make_env() -> AMRBicycleEnv:
    return AMRBicycleEnv(get_map_spec("forest_a"), max_steps=1200)


class TestV7P2MarkovObsPrevA(unittest.TestCase):
    def test_bicycle_obs_includes_prev_accel_scalar(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=7)

        expected_dim = 11 + int(env.obs_map_size) * int(env.obs_map_size)
        self.assertEqual(int(obs.shape[0]), int(expected_dim))

        prev_a_idx = 8
        self.assertAlmostEqual(float(obs[prev_a_idx]), 0.0, places=6)

        action = int(np.argmax(env.action_table[:, 1]))
        a_cmd = float(env.action_table[action, 1])
        obs_next, _, _, _, _ = env.step(action)
        expected_prev_a_n = float(np.clip(a_cmd / float(env.model.a_max_m_s2), -1.0, 1.0))
        self.assertAlmostEqual(float(obs_next[prev_a_idx]), float(expected_prev_a_n), places=6)

    def test_infer_flat_obs_cnn_layout_accepts_bicycle_11_plus_n2(self) -> None:
        layout = infer_flat_obs_cnn_layout(11 + 12 * 12)
        self.assertEqual(int(layout.scalar_dim), 11)
        self.assertEqual(int(layout.map_channels), 1)
        self.assertEqual(int(layout.map_size), 12)


if __name__ == "__main__":
    unittest.main()
