from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def path_length(path: list[tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:], strict=False):
        total += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    return float(total)


def corner_angles_deg(path: list[tuple[float, float]]) -> list[float]:
    if len(path) < 3:
        return []
    angles: list[float] = []
    for (x0, y0), (x1, y1), (x2, y2) in zip(path[:-2], path[1:-1], path[2:], strict=False):
        v1 = (float(x1) - float(x0), float(y1) - float(y0))
        v2 = (float(x2) - float(x1), float(y2) - float(y1))
        if abs(v1[0]) + abs(v1[1]) < 1e-9 or abs(v2[0]) + abs(v2[1]) < 1e-9:
            continue
        dot = float(v1[0] * v2[0] + v1[1] * v2[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cos = max(-1.0, min(1.0, dot / (n1 * n2)))
        ang = float(math.degrees(math.acos(cos)))
        angles.append(ang)
    return angles


def num_path_corners(path: list[tuple[float, float]], *, angle_threshold_deg: float = 1.0) -> int:
    th = float(angle_threshold_deg)
    if th < 0:
        raise ValueError("angle_threshold_deg must be >= 0")
    return int(sum(1 for a in corner_angles_deg(path) if a >= th))


def max_corner_degree(path: list[tuple[float, float]]) -> float:
    angles = corner_angles_deg(path)
    return float(max(angles) if angles else 0.0)


def min_distance_to_obstacle(grid: np.ndarray, path: list[tuple[float, float]]) -> float:
    """Approximate minimum Euclidean clearance from path vertices to obstacle boundaries (cell units)."""
    if len(path) == 0:
        return 0.0
    h, w = grid.shape

    # cv2.distanceTransform expects origin at top-left, but Euclidean distances are invariant
    # under vertical flips. We compute in top-origin then flip back to y=0 bottom for sampling.
    grid_top = grid[::-1, :]
    free = (grid_top == 0).astype(np.uint8) * 255
    dist_top = cv2.distanceTransform(
        free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
    ).astype(np.float32)

    # Convert center-to-center distances to approximate clearance from the path vertex
    # (assumed at the cell center) to the obstacle cell boundary.
    dist_top = np.maximum(0.0, dist_top - 0.5).astype(np.float32, copy=False)
    dist = dist_top[::-1, :]

    def sample_bilinear(x: float, y: float) -> float:
        if not (0.0 <= x <= (w - 1) and 0.0 <= y <= (h - 1)):
            return float("inf")
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        fx = float(x - x0)
        fy = float(y - y0)
        v00 = float(dist[y0, x0])
        v10 = float(dist[y0, x1])
        v01 = float(dist[y1, x0])
        v11 = float(dist[y1, x1])
        v0 = v00 * (1.0 - fx) + v10 * fx
        v1 = v01 * (1.0 - fx) + v11 * fx
        return v0 * (1.0 - fy) + v1 * fy

    best = float("inf")
    for x, y in path:
        best = min(best, sample_bilinear(float(x), float(y)))
    return float(best if math.isfinite(best) else 0.0)


@dataclass(frozen=True)
class KPI:
    avg_path_length: float
    num_corners: float
    min_collision_dist: float
    inference_time_s: float
    max_corner_deg: float
