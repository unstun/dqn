from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from forest_vehicle_dqn.agents import AgentConfig, DQNFamilyAgent
from forest_vehicle_dqn.networks import CNNQNetwork, infer_flat_obs_cnn_layout


def test_cnnqnetwork_backbones_output_shape() -> None:
    obs_dim = 10 + 12 * 12
    n_actions = 225
    layout = infer_flat_obs_cnn_layout(obs_dim)

    for backbone in ("legacy", "globalcnn"):
        net = CNNQNetwork(
            obs_dim,
            n_actions,
            scalar_dim=layout.scalar_dim,
            map_channels=layout.map_channels,
            map_size=layout.map_size,
            hidden_dim=64,
            hidden_layers=2,
            cnn_backbone=backbone,
            globalcnn_width=16,
            globalcnn_dropout=0.1,
        )
        x = torch.randn(4, obs_dim, dtype=torch.float32)
        y = net(x)
        assert y.shape == (4, n_actions)


def test_dqn_agent_globalcnn_save_load_roundtrip(tmp_path) -> None:
    obs_dim = 10 + 12 * 12
    n_actions = 225
    cfg = replace(
        AgentConfig(),
        hidden_dim=64,
        hidden_layers=2,
        cnn_backbone="globalcnn",
        globalcnn_width=16,
        globalcnn_dropout=0.1,
    )

    agent = DQNFamilyAgent("cnn-ddqn", obs_dim, n_actions, config=cfg, seed=0, device="cpu")
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.act(obs, episode=0, explore=False)
    assert 0 <= int(action) < n_actions

    ckpt = tmp_path / "cnn_ddqn_globalcnn.pt"
    agent.save(ckpt)

    loaded = DQNFamilyAgent("cnn-ddqn", obs_dim, n_actions, config=AgentConfig(), seed=1, device="cpu")
    loaded.load(ckpt)

    assert loaded.arch == "cnn"
    assert str(loaded._net_kwargs.get("cnn_backbone")) == "globalcnn"
    q = loaded.q(torch.from_numpy(obs).unsqueeze(0))
    assert tuple(q.shape) == (1, n_actions)


def test_legacy_checkpoint_without_globalcnn_keys_still_loads(tmp_path) -> None:
    obs_dim = 10 + 12 * 12
    n_actions = 225
    old_cfg = replace(AgentConfig(), hidden_dim=64, hidden_layers=2, cnn_backbone="legacy")
    old_agent = DQNFamilyAgent("cnn-ddqn", obs_dim, n_actions, config=old_cfg, seed=2, device="cpu")

    old_ckpt = tmp_path / "legacy_old_format.pt"
    old_agent.save(old_ckpt)

    payload = torch.load(old_ckpt, map_location="cpu")
    nk = dict(payload.get("network_kwargs") or {})
    payload["network_kwargs"] = {
        "scalar_dim": int(nk["scalar_dim"]),
        "map_channels": int(nk["map_channels"]),
        "map_size": int(nk["map_size"]),
    }
    torch.save(payload, old_ckpt)

    loaded = DQNFamilyAgent("cnn-ddqn", obs_dim, n_actions, config=AgentConfig(), seed=3, device="cpu")
    loaded.load(old_ckpt)
    assert str(loaded._net_kwargs.get("cnn_backbone")) == "legacy"

