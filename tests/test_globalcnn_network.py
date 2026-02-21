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

    for backbone in ("legacy", "globalcnn", "globalcnn_fusion"):
        kwargs: dict[str, object] = {}
        if backbone != "legacy":
            kwargs = {
                "globalcnn_spatial_prior": True,
                "globalcnn_prior_sigma": 0.2,
                "globalcnn_fusion_layernorm": (backbone == "globalcnn_fusion"),
                "globalcnn_fusion_layernorm_eps": 1e-5,
            }
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
            **kwargs,
        )
        x = torch.randn(4, obs_dim, dtype=torch.float32)
        y = net(x)
        assert y.shape == (4, n_actions)


def test_globalcnn_spatial_prior_channel_wiring() -> None:
    obs_dim = 10 + 12 * 12
    n_actions = 225
    layout = infer_flat_obs_cnn_layout(obs_dim)

    net_prior = CNNQNetwork(
        obs_dim,
        n_actions,
        scalar_dim=layout.scalar_dim,
        map_channels=layout.map_channels,
        map_size=layout.map_size,
        hidden_dim=64,
        hidden_layers=2,
        cnn_backbone="globalcnn_fusion",
        globalcnn_width=16,
        globalcnn_dropout=0.0,
        globalcnn_spatial_prior=True,
        globalcnn_prior_sigma=0.2,
        globalcnn_fusion_layernorm=True,
        globalcnn_fusion_layernorm_eps=1e-5,
    )
    assert net_prior._global_blocks is not None
    assert net_prior._fusion_local_branch is not None
    assert isinstance(net_prior._global_blocks[0][0], torch.nn.Conv2d)
    assert isinstance(net_prior._fusion_local_branch[0], torch.nn.Conv2d)
    assert isinstance(net_prior._fusion_norm, torch.nn.LayerNorm)
    assert net_prior._global_blocks[0][0].in_channels == layout.map_channels + 2
    assert net_prior._fusion_local_branch[0].in_channels == layout.map_channels + 2

    net_no_prior = CNNQNetwork(
        obs_dim,
        n_actions,
        scalar_dim=layout.scalar_dim,
        map_channels=layout.map_channels,
        map_size=layout.map_size,
        hidden_dim=64,
        hidden_layers=2,
        cnn_backbone="globalcnn_fusion",
        globalcnn_width=16,
        globalcnn_dropout=0.0,
        globalcnn_spatial_prior=False,
        globalcnn_fusion_layernorm=False,
    )
    assert net_no_prior._global_blocks is not None
    assert net_no_prior._fusion_local_branch is not None
    assert isinstance(net_no_prior._global_blocks[0][0], torch.nn.Conv2d)
    assert isinstance(net_no_prior._fusion_local_branch[0], torch.nn.Conv2d)
    assert net_no_prior._fusion_norm is None
    assert net_no_prior._global_blocks[0][0].in_channels == layout.map_channels
    assert net_no_prior._fusion_local_branch[0].in_channels == layout.map_channels


def test_dqn_agent_globalcnn_save_load_roundtrip(tmp_path) -> None:
    obs_dim = 10 + 12 * 12
    n_actions = 225
    for backbone in ("globalcnn", "globalcnn_fusion"):
        cfg = replace(
            AgentConfig(),
            hidden_dim=64,
            hidden_layers=2,
            cnn_backbone=backbone,
            globalcnn_width=16,
            globalcnn_dropout=0.1,
            globalcnn_spatial_prior=True,
            globalcnn_prior_sigma=0.2,
            globalcnn_fusion_layernorm=(backbone == "globalcnn_fusion"),
            globalcnn_fusion_layernorm_eps=1e-5,
        )

        agent = DQNFamilyAgent("cnn-ddqn", obs_dim, n_actions, config=cfg, seed=0, device="cpu")
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.act(obs, episode=0, explore=False)
        assert 0 <= int(action) < n_actions

        ckpt = tmp_path / f"cnn_ddqn_{backbone}.pt"
        agent.save(ckpt)

        loaded = DQNFamilyAgent("cnn-ddqn", obs_dim, n_actions, config=AgentConfig(), seed=1, device="cpu")
        loaded.load(ckpt)

        assert loaded.arch == "cnn"
        assert str(loaded._net_kwargs.get("cnn_backbone")) == backbone
        assert bool(loaded._net_kwargs.get("globalcnn_spatial_prior")) is True
        assert float(loaded._net_kwargs.get("globalcnn_prior_sigma")) == 0.2
        assert bool(loaded._net_kwargs.get("globalcnn_fusion_layernorm")) is (backbone == "globalcnn_fusion")
        assert float(loaded._net_kwargs.get("globalcnn_fusion_layernorm_eps")) == 1e-5
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
    assert bool(loaded._net_kwargs.get("globalcnn_spatial_prior", False)) is False
    assert bool(loaded._net_kwargs.get("globalcnn_fusion_layernorm", False)) is False
