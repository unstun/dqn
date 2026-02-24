from __future__ import annotations

import numpy as np
import torch

from forest_vehicle_dqn.networks import CNNQNetwork, MLPQNetwork


def test_mlp_dueling_forward_shape_and_finite() -> None:
    torch.manual_seed(0)
    input_dim = 11
    n_actions = 17
    batch = 4
    x = torch.randn((batch, input_dim), dtype=torch.float32)

    plain = MLPQNetwork(input_dim, n_actions, hidden_dim=64, hidden_layers=2, dueling=False)
    y_plain = plain(x)
    assert tuple(y_plain.shape) == (batch, n_actions)
    assert torch.isfinite(y_plain).all()

    dueling = MLPQNetwork(input_dim, n_actions, hidden_dim=64, hidden_layers=2, dueling=True, dueling_hidden_dim=32)
    y_duel = dueling(x)
    assert tuple(y_duel.shape) == (batch, n_actions)
    assert torch.isfinite(y_duel).all()


def test_cnn_dueling_forward_shape_and_finite() -> None:
    torch.manual_seed(0)
    scalar_dim = 10
    map_channels = 1
    map_size = 6
    input_dim = int(scalar_dim + map_channels * map_size * map_size)
    n_actions = 11
    batch = 3

    x = torch.randn((batch, input_dim), dtype=torch.float32)

    plain = CNNQNetwork(
        input_dim,
        n_actions,
        scalar_dim=scalar_dim,
        map_channels=map_channels,
        map_size=map_size,
        hidden_dim=64,
        hidden_layers=2,
        cnn_backbone="legacy",
        dueling=False,
    )
    y_plain = plain(x)
    assert tuple(y_plain.shape) == (batch, n_actions)
    assert torch.isfinite(y_plain).all()

    dueling = CNNQNetwork(
        input_dim,
        n_actions,
        scalar_dim=scalar_dim,
        map_channels=map_channels,
        map_size=map_size,
        hidden_dim=64,
        hidden_layers=2,
        cnn_backbone="legacy",
        dueling=True,
        dueling_hidden_dim=32,
    )
    y_duel = dueling(x)
    assert tuple(y_duel.shape) == (batch, n_actions)
    assert torch.isfinite(y_duel).all()

