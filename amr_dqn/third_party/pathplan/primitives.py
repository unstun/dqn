from dataclasses import dataclass
from typing import List

from .robot import AckermannParams


@dataclass(frozen=True)
class MotionPrimitive:
    steering: float  # radians
    direction: int  # +1 forward, -1 reverse
    step: float  # meters
    weight: float = 1.0  # cost multiplier


def default_primitives(
    params: AckermannParams,
    step_length: float = 0.3,
    delta_scale: float = 0.5,
    steer_bins: int = 3,
) -> List[MotionPrimitive]:
    """Generate discrete steering-rate primitives for Hybrid A*.

    By default this matches the common (paper-style) action set:
        steer_bins=3 -> {max-left, straight, max-right} x {forward, reverse} = 6 actions.

    Set steer_bins=5 to recover the previous behavior:
        {-max, -small, 0, +small, +max} x {forward, reverse} = 10 actions
    where `small = delta_scale * max`.
    """

    n = int(steer_bins)
    if n < 3:
        raise ValueError("steer_bins must be >= 3")
    if n % 2 == 0:
        raise ValueError("steer_bins must be odd so 0-steer is included")

    delta_max = float(params.max_steer)
    if n == 3:
        steering_bins = (-delta_max, 0.0, +delta_max)
    elif n == 5:
        delta_small = float(delta_scale) * delta_max
        steering_bins = (-delta_max, -delta_small, 0.0, +delta_small, +delta_max)
    else:
        step = (2.0 * delta_max) / float(n - 1)
        steering_bins = tuple(-delta_max + step * i for i in range(n))

    prims: List[MotionPrimitive] = []
    for steer in steering_bins:
        prims.append(MotionPrimitive(float(steer), +1, float(step_length), weight=1.0))
    for steer in steering_bins:
        prims.append(MotionPrimitive(float(steer), -1, float(step_length), weight=1.2))  # slight reverse penalty
    return prims


def primitive_cost(primitive: MotionPrimitive) -> float:
    """Base traversal cost for one primitive."""
    return abs(primitive.step) * primitive.weight
