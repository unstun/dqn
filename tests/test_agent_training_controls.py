from __future__ import annotations

from dataclasses import replace

import numpy as np

from forest_vehicle_dqn.agents import AgentConfig, DQNFamilyAgent


def _agent_cfg(
    *,
    batch_size: int = 1,
    reward_scale: float = 1.0,
    reward_clip_abs: float = 0.0,
    grad_clip_norm: float = 10.0,
    demo_mode: str = "dqfd",
) -> AgentConfig:
    return replace(
        AgentConfig(),
        hidden_layers=2,
        hidden_dim=64,
        replay_capacity=128,
        batch_size=int(batch_size),
        n_step=1,
        replay_prioritized=False,
        replay_stratified=False,
        reward_scale=float(reward_scale),
        reward_clip_abs=float(reward_clip_abs),
        grad_clip_norm=float(grad_clip_norm),
        demo_mode=str(demo_mode),
    )


def _make_agent(cfg: AgentConfig) -> DQNFamilyAgent:
    return DQNFamilyAgent("mlp-ddqn", obs_dim=8, n_actions=4, config=cfg, seed=0, device="cpu")


def _dummy_transition(i: int) -> tuple[np.ndarray, int, float, np.ndarray]:
    obs = np.full((8,), float(i), dtype=np.float32)
    next_obs = np.full((8,), float(i + 1), dtype=np.float32)
    action = int(i % 4)
    reward = float(1000.0 if (i % 2 == 0) else -1000.0)
    return obs, action, reward, next_obs


def test_reward_scale_and_clip_are_applied_before_replay_store() -> None:
    agent = _make_agent(_agent_cfg(reward_scale=0.1, reward_clip_abs=0.5))
    obs, action, _, next_obs = _dummy_transition(0)
    agent.observe(obs, action, 20.0, next_obs, False, next_action_mask=np.ones((4,), dtype=bool))
    batch = agent.replay.sample(1)
    assert np.isclose(float(batch.rewards_1[0]), 0.5, atol=1e-6)
    assert np.isclose(float(batch.rewards_n[0]), 0.5, atol=1e-6)


def test_reward_scale_without_clip_keeps_signed_value() -> None:
    agent = _make_agent(_agent_cfg(reward_scale=0.1, reward_clip_abs=0.0))
    obs, action, _, next_obs = _dummy_transition(1)
    agent.observe(obs, action, -20.0, next_obs, False, next_action_mask=np.ones((4,), dtype=bool))
    batch = agent.replay.sample(1)
    assert np.isclose(float(batch.rewards_1[0]), -2.0, atol=1e-6)
    assert np.isclose(float(batch.rewards_n[0]), -2.0, atol=1e-6)


def test_update_returns_grad_clip_observability_metrics() -> None:
    agent = _make_agent(_agent_cfg(batch_size=2, grad_clip_norm=1e-6, demo_mode="legacy"))
    for i in range(6):
        obs, action, reward, next_obs = _dummy_transition(i)
        agent.observe(obs, action, reward, next_obs, False, next_action_mask=np.ones((4,), dtype=bool))

    out = agent.update()
    assert out
    assert "grad_norm_pre_clip" in out
    assert "grad_clip_hit" in out
    assert np.isfinite(float(out["grad_norm_pre_clip"]))
    assert float(out["grad_norm_pre_clip"]) >= 0.0
    assert float(out["grad_clip_hit"]) in (0.0, 1.0)


def test_pretrain_tracks_last_grad_stats() -> None:
    agent = _make_agent(_agent_cfg(batch_size=2, grad_clip_norm=1.0, demo_mode="legacy"))
    for i in range(6):
        obs, action, reward, next_obs = _dummy_transition(i)
        agent.observe(
            obs,
            action,
            reward,
            next_obs,
            False,
            demo=True,
            next_action_mask=np.ones((4,), dtype=bool),
        )

    trained = int(agent.pretrain_on_demos(steps=5))
    assert trained >= 1
    stats = dict(getattr(agent, "_last_pretrain_stats", {}))
    assert "grad_norm_pre_clip_mean" in stats
    assert "grad_clip_hit_rate" in stats
    assert np.isfinite(float(stats["grad_norm_pre_clip_mean"]))
    assert 0.0 <= float(stats["grad_clip_hit_rate"]) <= 1.0
