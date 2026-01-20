from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MapSpec(Protocol):
    name: str
    start_xy: tuple[int, int]
    goal_xy: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]: ...

    def obstacle_grid(self) -> np.ndarray: ...


@dataclass(frozen=True)
class GridMapSpec:
    name: str
    rows_y0_bottom: list[str]
    start_xy: tuple[int, int]
    goal_xy: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        height = len(self.rows_y0_bottom)
        width = len(self.rows_y0_bottom[0]) if height else 0
        return width, height

    def obstacle_grid(self) -> np.ndarray:
        """Returns a (H, W) uint8 array with y=0 at bottom, 1=obstacle."""
        width, height = self.size
        if height == 0 or width == 0:
            raise ValueError(f"Empty map: {self.name!r}")
        if any(len(r) != width for r in self.rows_y0_bottom):
            raise ValueError(f"Non-rectangular map: {self.name!r}")

        grid = np.zeros((height, width), dtype=np.uint8)
        for y, row in enumerate(self.rows_y0_bottom):
            for x, ch in enumerate(row):
                if ch == "#":
                    grid[y, x] = 1
                elif ch == ".":
                    continue
                else:
                    raise ValueError(
                        f"Invalid char {ch!r} in map {self.name!r} at (x={x}, y={y})"
                    )
        return grid


@dataclass(frozen=True)
class ArrayGridMapSpec:
    name: str
    grid_y0_bottom: np.ndarray  # (H, W) uint8 with y=0 bottom, 1=obstacle
    start_xy: tuple[int, int]
    goal_xy: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.grid_y0_bottom.shape
        return int(w), int(h)

    def obstacle_grid(self) -> np.ndarray:
        return self.grid_y0_bottom.astype(np.uint8, copy=True)


ENV_ORDER: tuple[str, ...] = ("a", "b", "c", "d")
FOREST_ENV_ORDER: tuple[str, ...] = ("forest_a", "forest_b", "forest_c", "forest_d")


_MAP_A = [
    "......##......................",
    "....###..........######.......",
    "...###...........#............",
    "..##.............#............",
    "......####.......######.......",
    "......#..#....................",
    "......#..#.......#.#..........",
    "......#..#....................",
    ".....##..#......####..........",
    "......#..#......####..........",
    "......#..#.......##...........",
    "..............................",
    "....................###.......",
    "...................#####......",
    "..............................",
    ".................#............",
    "................###...........",
    "..######.......#####..........",
    "..######........###...........",
    "..######.........#............",
    "........................#####.",
    "............##..........#####.",
    ".............##.........#####.",
    "........#.....##..............",
    ".......##......##.............",
    "......###...........##........",
    ".....####..........####.......",
    "....#####.........######......",
    "...######........########.....",
    "..#######.....................",
]

_MAP_B = [
    "...#..........................",
    "..........##.....##...........",
    "..........##.....##...........",
    "..........##.....##...........",
    "..............................",
    "..............................",
    "..............................",
    "....................#######...",
    ".#####..............########..",
    ".#####...............########.",
    ".#####................#######.",
    ".#####.................#######",
    ".#####..................######",
    "...........########......#####",
    "...........########...........",
    "...........########...........",
    "..............................",
    "####..........................",
    "#####.........................",
    ".#####........................",
    "..#####.......................",
    ".....................#........",
    "..........#####......#........",
    "..........#####......#........",
    ".#####....#####......#........",
    ".#####....#####......#........",
    ".#####....#####......#########",
    ".#####...............#########",
    ".#####........................",
    "..............................",
]

_MAP_C = [
    "...#..........................",
    "..........##....##............",
    "..........##....##............",
    "..........##....##............",
    "..............................",
    "..............................",
    "..............................",
    "...................#######....",
    ".#####.............########...",
    ".#####..............########..",
    ".#####..............#########.",
    ".#####................#######.",
    ".#####..................######",
    "...........#######......######",
    "...........#######............",
    "...........#######............",
    "..............................",
    "####..........................",
    "#####.........................",
    ".#####.............##....#....",
    "..#####............##...##....",
    "...................#######....",
    "..........####......######....",
    "..........####................",
    ".#####....####................",
    ".#####....####...#.......#....",
    ".#####....####...#......##....",
    ".#####.#.........##.#...##....",
    ".#####...........#..#...##....",
    ".................#########....",
]

_MAP_D = [
    ".................###..........",
    ".................###..........",
    ".................###..........",
    ".................###..........",
    "..........##.....##...........",
    "..###....###..................",
    "..###...####..................",
    "..###...####........######....",
    "..###....###.............#....",
    "..###.....##.............#....",
    "...#....................##....",
    "..###...................##....",
    "..###...........#.......##....",
    "...###.#...######....#..######",
    "....###.....#####.............",
    ".....###.....###..............",
    "####...##.....................",
    "..##..................##......",
    "..##................######....",
    "..##...............###..###...",
    "..#####...........####...###..",
    "..#####..........####.....##..",
    ".................###.......##.",
    "..######.....#................",
    "..##........###...#########...",
    "..######....###...#.......#...",
    ".............#....#.......#...",
    "....##............##......#...",
    "....##............##......#...",
    "..####............#########...",
]


MAPS: dict[str, GridMapSpec] = {
    "a": GridMapSpec(name="a", rows_y0_bottom=_MAP_A, start_xy=(17, 7), goal_xy=(10, 28)),
    "b": GridMapSpec(name="b", rows_y0_bottom=_MAP_B, start_xy=(4, 0), goal_xy=(25, 29)),
    "c": GridMapSpec(name="c", rows_y0_bottom=_MAP_C, start_xy=(4, 0), goal_xy=(29, 29)),
    "d": GridMapSpec(name="d", rows_y0_bottom=_MAP_D, start_xy=(4, 0), goal_xy=(13, 29)),
}


_FOREST_CACHE: dict[str, ArrayGridMapSpec] = {}


def _get_forest_spec(env_name: str) -> ArrayGridMapSpec:
    if env_name in _FOREST_CACHE:
        return _FOREST_CACHE[env_name]

    from amr_dqn.maps.forest import ForestParams, generate_forest_grid

    if env_name == "forest_a":
        # Large map, wide gaps.
        seed = 101
        params = ForestParams(
            width_cells=160,
            height_cells=160,
            trunk_count=80,
            trunk_gap_m=1.30,
            bush_cluster_count=0,
        )
    elif env_name == "forest_b":
        # Small map, tighter gaps (must remain bicycle-feasible).
        seed = 202
        params = ForestParams(
            width_cells=96,
            height_cells=96,
            trunk_count=28,
            trunk_gap_m=1.35,
            bush_cluster_count=0,
        )
    elif env_name == "forest_c":
        # Large map, denser layout (must remain bicycle-feasible).
        seed = 303
        params = ForestParams(
            width_cells=160,
            height_cells=160,
            trunk_count=85,
            trunk_gap_m=1.25,
            bush_cluster_count=0,
        )
    elif env_name == "forest_d":
        # Small map, wide gaps.
        seed = 404
        params = ForestParams(
            width_cells=96,
            height_cells=96,
            trunk_count=28,
            trunk_gap_m=1.30,
            bush_cluster_count=0,
        )
    else:
        raise KeyError(env_name)

    # Footprint clearance for reachability checks: (r + safe_distance + eps_cell).
    # This ensures the generated forest maps are not only collision-free but also have
    # enough clearance for the reward's safe-distance threshold.
    # r=0.436m for the two-circle approximation, eps_cell=sqrt(2)/2*cell_size.
    r_m = 0.436
    eps_cell_m = (2.0**0.5) * 0.5 * float(params.cell_size_m)
    safe_distance_m = 0.20
    footprint_clearance_m = r_m + safe_distance_m + eps_cell_m

    grid, start_xy, goal_xy = generate_forest_grid(
        params=params,
        rng=np.random.default_rng(seed),
        footprint_clearance_m=footprint_clearance_m,
    )
    spec = ArrayGridMapSpec(name=env_name, grid_y0_bottom=grid, start_xy=start_xy, goal_xy=goal_xy)
    _FOREST_CACHE[env_name] = spec
    return spec


def get_map_spec(env_name: str) -> MapSpec:
    if env_name in FOREST_ENV_ORDER:
        return _get_forest_spec(env_name)
    try:
        return MAPS[env_name]
    except KeyError as e:
        raise KeyError(
            f"Unknown env {env_name!r}. Options: {sorted(list(MAPS) + list(FOREST_ENV_ORDER))}"
        ) from e
