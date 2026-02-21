from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int = 128, hidden_layers: int = 2):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Backwards-compatible name (historically this repo only had an MLP Q-network).
QNetwork = MLPQNetwork


@dataclass(frozen=True)
class FlatObsCnnLayout:
    scalar_dim: int
    map_channels: int
    map_size: int


def infer_flat_obs_cnn_layout(obs_dim: int) -> FlatObsCnnLayout:
    """Infer (scalar_dim, map_channels, map_size) for this repo's flat observations.

    Supported layouts:
    - AMRGridEnv:   obs = [5 scalars] + [1 * (N*N) map]
    - AMRBicycleEnv:obs = [10 scalars] + [1 * (N*N) map]  (occ)
    """

    d = int(obs_dim)
    if d <= 0:
        raise ValueError("obs_dim must be > 0")

    candidates: list[FlatObsCnnLayout] = []
    for scalar_dim, channels in ((5, 1), (10, 1)):
        rem = d - int(scalar_dim)
        if rem <= 0:
            continue
        if rem % int(channels) != 0:
            continue
        per = rem // int(channels)
        n = int(round(math.sqrt(per)))
        if n > 0 and n * n == per:
            candidates.append(FlatObsCnnLayout(scalar_dim=int(scalar_dim), map_channels=int(channels), map_size=int(n)))

    if not candidates:
        raise ValueError(
            f"Cannot infer CNN layout from obs_dim={d}. Expected 5+N^2 (grid) or 10+N^2 (bicycle)."
        )
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous CNN layout for obs_dim={d}: {candidates}")
    return candidates[0]


class CNNQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        scalar_dim: int,
        map_channels: int,
        map_size: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        cnn_backbone: str = "legacy",
        globalcnn_width: int = 32,
        globalcnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.scalar_dim = int(scalar_dim)
        self.map_channels = int(map_channels)
        self.map_size = int(map_size)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.cnn_backbone = str(cnn_backbone).lower().strip()
        self.globalcnn_width = int(globalcnn_width)
        self.globalcnn_dropout = float(globalcnn_dropout)

        if self.scalar_dim < 0:
            raise ValueError("scalar_dim must be >= 0")
        if self.map_channels < 1:
            raise ValueError("map_channels must be >= 1")
        if self.map_size < 1:
            raise ValueError("map_size must be >= 1")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        if self.globalcnn_width < 8:
            raise ValueError("globalcnn_width must be >= 8")
        if not (0.0 <= self.globalcnn_dropout < 1.0):
            raise ValueError("globalcnn_dropout must be in [0,1)")

        expected = int(self.scalar_dim) + int(self.map_channels) * int(self.map_size) * int(self.map_size)
        if int(input_dim) != expected:
            raise ValueError(
                f"CNNQNetwork expected input_dim={expected} (scalar_dim={self.scalar_dim}, "
                f"map_channels={self.map_channels}, map_size={self.map_size}), got {int(input_dim)}"
            )

        self.conv: nn.Sequential | None = None
        self._global_blocks: nn.ModuleList | None = None
        self._global_dropout: nn.Module | None = None
        conv_out_dim = 0

        if self.cnn_backbone == "legacy":
            # A real 2D CNN over the downsampled global maps. Designed for small maps (e.g. 12x12).
            self.conv = nn.Sequential(
                nn.Conv2d(self.map_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

            with torch.no_grad():
                dummy = torch.zeros((1, self.map_channels, self.map_size, self.map_size), dtype=torch.float32)
                conv_out = self.conv(dummy)
                conv_out_dim = int(conv_out.flatten(start_dim=1).shape[1])
        elif self.cnn_backbone == "globalcnn":
            w = int(self.globalcnn_width)
            self._global_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.map_channels, w, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(w, w, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(w, 2 * w, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(2 * w, 2 * w, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(2 * w, 4 * w, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(4 * w, 4 * w, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(4 * w, 4 * w, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(4 * w, 4 * w, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                    ),
                ]
            )
            self._global_dropout = (
                nn.Dropout(p=float(self.globalcnn_dropout))
                if float(self.globalcnn_dropout) > 0.0
                else nn.Identity()
            )
            # avg pool + max pool for each stage.
            conv_out_dim = int(2 * (w + 2 * w + 4 * w + 4 * w))
        else:
            raise ValueError("cnn_backbone must be one of: legacy, globalcnn")

        fc_in_dim = int(self.scalar_dim) + int(conv_out_dim)

        layers: list[nn.Module] = []
        layers.append(nn.Linear(fc_in_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        for _ in range(int(hidden_layers) - 1):
            layers.append(nn.Linear(int(hidden_dim), int(hidden_dim)))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(int(hidden_dim), int(output_dim)))
        self.head = nn.Sequential(*layers)

    def _forward_globalcnn(self, maps: torch.Tensor) -> torch.Tensor:
        if self._global_blocks is None:
            raise RuntimeError("globalcnn backbone blocks are not initialized")

        feats: list[torch.Tensor] = []
        x = maps
        for block in self._global_blocks:
            x = block(x)
            avg = F.adaptive_avg_pool2d(x, output_size=1).flatten(start_dim=1)
            mx = F.adaptive_max_pool2d(x, output_size=1).flatten(start_dim=1)
            feats.append(torch.cat([avg, mx], dim=1))

        out = torch.cat(feats, dim=1)
        if self._global_dropout is not None:
            out = self._global_dropout(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError("CNNQNetwork expects (batch, obs_dim) input")
        if int(x.shape[1]) != int(self.input_dim):
            raise ValueError(f"CNNQNetwork expected input_dim={self.input_dim}, got {int(x.shape[1])}")

        scalars = x[:, : self.scalar_dim]
        maps_flat = x[:, self.scalar_dim :]
        maps = maps_flat.reshape(int(x.shape[0]), self.map_channels, self.map_size, self.map_size)
        if self.cnn_backbone == "legacy":
            if self.conv is None:
                raise RuntimeError("legacy cnn backbone is not initialized")
            conv = self.conv(maps)
            map_feat = conv.flatten(start_dim=1)
        else:
            map_feat = self._forward_globalcnn(maps)
        feats = torch.cat([scalars, map_feat], dim=1)
        return self.head(feats)
