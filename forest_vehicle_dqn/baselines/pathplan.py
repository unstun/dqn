from __future__ import annotations

import math
import time
from heapq import heappop, heappush
from dataclasses import dataclass
from typing import Any

import numpy as np
from forest_vehicle_dqn.smoothing import chaikin_smooth

from forest_vehicle_dqn.third_party.pathplan import (
    AckermannParams,
    AckermannState,
    GridMap,
    HybridAStarPlanner,
    OrientedBoxFootprint,
    RRTStarPlanner,
    TwoCircleFootprint,
)
from forest_vehicle_dqn.third_party.pathplan.geometry import GridFootprintChecker


@dataclass(frozen=True)
class PlannerResult:
    path_xy_cells: list[tuple[float, float]]
    # Optional per-point heading (radians) for the returned path. When present, this is the
    # planner's internal heading state for each (x,y) waypoint (e.g., Hybrid A* AckermannState.theta).
    path_theta_rad: list[float] | None
    time_s: float
    success: bool
    stats: dict[str, Any]


@dataclass(frozen=True)
class AStarCurveOptConfig:
    enable: bool = True
    max_replans: int = 200
    resample_ds_m: float = 0.2
    collision_step_m: float = 0.1
    shortcut_passes: int = 2


def _path_headings(path_xy_m: np.ndarray) -> np.ndarray:
    n = int(path_xy_m.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    h = np.zeros((n,), dtype=np.float64)
    dxy = np.diff(path_xy_m, axis=0)
    seg_h = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float64, copy=False)
    h[:-1] = seg_h
    h[-1] = seg_h[-1]
    return h


def _resample_polyline(path_xy_m: np.ndarray, *, ds_m: float) -> np.ndarray:
    pts = np.asarray(path_xy_m, dtype=np.float64)
    if pts.ndim != 2 or int(pts.shape[0]) < 2:
        return pts.astype(np.float64, copy=False)

    ds = max(1e-6, float(ds_m))
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1]).astype(np.float64, copy=False)
    cum = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_len, dtype=np.float64)])
    total = float(cum[-1])
    if not math.isfinite(total) or total <= 1e-9:
        return pts.astype(np.float64, copy=False)

    n_samples = int(max(2, math.ceil(total / ds) + 1))
    targets = np.linspace(0.0, total, num=n_samples, dtype=np.float64)

    out = np.zeros((int(targets.size), 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while (j + 1) < int(cum.size) and float(cum[j + 1]) < float(t):
            j += 1
        if j >= int(seg_len.size):
            out[i] = pts[-1]
            continue
        l0 = float(cum[j])
        l1 = float(cum[j + 1])
        if not (l1 > l0):
            out[i] = pts[j]
            continue
        u = float((float(t) - l0) / (l1 - l0))
        out[i] = (1.0 - u) * pts[j] + u * pts[j + 1]

    out[0] = pts[0]
    out[-1] = pts[-1]
    return out.astype(np.float64, copy=False)


def _segment_collision_free(
    checker: GridFootprintChecker,
    *,
    p0_m: tuple[float, float],
    p1_m: tuple[float, float],
    collision_step_m: float,
) -> bool:
    x0, y0 = float(p0_m[0]), float(p0_m[1])
    x1, y1 = float(p1_m[0]), float(p1_m[1])
    theta = float(math.atan2(float(y1) - float(y0), float(x1) - float(x0)))
    return not bool(
        checker.motion_collides(
            (float(x0), float(y0), float(theta)),
            (float(x1), float(y1), float(theta)),
            step=max(1e-4, float(collision_step_m)),
        )
    )


def _curve_opt_search_inflation_margin_m(
    *,
    footprint: OrientedBoxFootprint | TwoCircleFootprint,
    params: AckermannParams,
) -> float:
    if isinstance(footprint, TwoCircleFootprint):
        footprint_margin = max(0.0, float(footprint.radius))
    elif isinstance(footprint, OrientedBoxFootprint):
        footprint_margin = max(
            0.0,
            float(math.hypot(float(footprint.half_length), float(footprint.half_width))),
        )
    else:
        footprint_margin = 0.0

    turn_margin = max(0.0, 0.3 * float(params.min_turn_radius))
    return max(float(footprint_margin), float(turn_margin))


def _shortcut_polyline(
    path_xy_m: np.ndarray,
    *,
    checker: GridFootprintChecker,
    collision_step_m: float,
    passes: int,
) -> np.ndarray:
    pts = np.asarray(path_xy_m, dtype=np.float64)
    if pts.ndim != 2 or int(pts.shape[0]) < 3:
        return pts.astype(np.float64, copy=False)

    cur = pts
    for _ in range(max(0, int(passes))):
        n = int(cur.shape[0])
        if n < 3:
            break
        out: list[np.ndarray] = [cur[0]]
        i = 0
        while i < n - 1:
            best_j = i + 1
            for j in range(n - 1, i + 1, -1):
                ok = _segment_collision_free(
                    checker,
                    p0_m=(float(cur[i, 0]), float(cur[i, 1])),
                    p1_m=(float(cur[j, 0]), float(cur[j, 1])),
                    collision_step_m=float(collision_step_m),
                )
                if ok:
                    best_j = int(j)
                    break
            out.append(cur[best_j])
            i = int(best_j)
        cur = np.asarray(out, dtype=np.float64)

    return cur.astype(np.float64, copy=False)


def _path_curvature_ok(*, path_xy_m: np.ndarray, kappa_max: float) -> bool:
    pts = np.asarray(path_xy_m, dtype=np.float64)
    n = int(pts.shape[0])
    if n < 3:
        return True

    kappa_thr = max(0.0, float(kappa_max))
    for i in range(1, n - 1):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[i + 1]

        a = float(np.hypot(*(p1 - p0)))
        b = float(np.hypot(*(p2 - p1)))
        c = float(np.hypot(*(p2 - p0)))
        if (a <= 1e-9) or (b <= 1e-9) or (c <= 1e-9):
            continue

        area2 = float(abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])))
        kappa = float(2.0 * area2 / max(1e-12, a * b * c))
        if not math.isfinite(kappa):
            return False
        if float(kappa) > float(kappa_thr) + 1e-6:
            return False
    return True


def _path_collision_free(
    *,
    checker: GridFootprintChecker,
    path_xy_m: np.ndarray,
    path_theta: np.ndarray,
    collision_step_m: float,
) -> bool:
    pts = np.asarray(path_xy_m, dtype=np.float64)
    th = np.asarray(path_theta, dtype=np.float64).reshape(-1)
    if pts.ndim != 2 or int(pts.shape[0]) < 2:
        return False
    if int(th.size) != int(pts.shape[0]):
        return False

    for i in range(int(pts.shape[0]) - 1):
        start = (float(pts[i, 0]), float(pts[i, 1]), float(th[i]))
        end = (float(pts[i + 1, 0]), float(pts[i + 1, 1]), float(th[i + 1]))
        if bool(checker.motion_collides(start, end, step=max(1e-4, float(collision_step_m)))):
            return False
    return True


def _optimize_grid_astar_path(
    *,
    grid_map: GridMap,
    footprint: OrientedBoxFootprint | TwoCircleFootprint,
    params: AckermannParams,
    path_xy_cells: list[tuple[float, float]],
    cfg: AStarCurveOptConfig,
) -> tuple[list[tuple[float, float]], list[float]] | None:
    if len(path_xy_cells) < 2:
        return None

    cell_size_m = float(grid_map.resolution)
    raw = np.asarray([(float(x) * cell_size_m, float(y) * cell_size_m) for x, y in path_xy_cells], dtype=np.float64)
    if raw.ndim != 2 or int(raw.shape[0]) < 2:
        return None

    checker = GridFootprintChecker(grid_map, footprint, theta_bins=72)
    step_m = max(1e-4, float(cfg.collision_step_m))

    short = _shortcut_polyline(
        raw,
        checker=checker,
        collision_step_m=step_m,
        passes=max(0, int(cfg.shortcut_passes)),
    )
    if short.ndim != 2 or int(short.shape[0]) < 2:
        return None

    ds_m = max(1e-6, float(cfg.resample_ds_m))
    dense = _resample_polyline(short, ds_m=ds_m)
    if dense.ndim != 2 or int(dense.shape[0]) < 2:
        return None

    # Corner-rounding on top of shortcut path.
    sm = chaikin_smooth(dense.astype(np.float32, copy=False), iterations=2).astype(np.float64, copy=False)
    if sm.ndim != 2 or int(sm.shape[0]) < 2:
        return None
    sm[0] = raw[0]
    sm[-1] = raw[-1]
    sm = _resample_polyline(sm, ds_m=ds_m)
    if sm.ndim != 2 or int(sm.shape[0]) < 2:
        return None

    headings = _path_headings(sm)
    min_turn = max(1e-9, float(params.min_turn_radius))
    kappa_max = 1.0 / float(min_turn)

    if not bool(_path_curvature_ok(path_xy_m=sm, kappa_max=kappa_max)):
        return None
    if not bool(_path_collision_free(checker=checker, path_xy_m=sm, path_theta=headings, collision_step_m=step_m)):
        return None

    out_cells = [(float(x) / cell_size_m, float(y) / cell_size_m) for x, y in sm]
    out_theta = [float(t) for t in headings]
    return out_cells, out_theta


def _plan_grid_astar_once(
    *,
    grid_map: GridMap,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    timeout_s: float,
    max_expanded: int,
    neighbors: tuple[tuple[int, int, float], ...],
) -> PlannerResult:
    cell_size_m = float(grid_map.resolution)
    start = (int(start_xy[0]), int(start_xy[1]))
    goal = (int(goal_xy[0]), int(goal_xy[1]))

    t0 = time.perf_counter()

    if not grid_map.in_bounds(start[0], start[1]) or not grid_map.in_bounds(goal[0], goal[1]):
        dt = float(time.perf_counter() - t0)
        return PlannerResult(
            path_xy_cells=[(float(start[0]), float(start[1]))],
            path_theta_rad=None,
            time_s=dt,
            success=False,
            stats={
                "expanded_nodes": 0,
                "path_cost": float("inf"),
                "failure_reason": "out_of_bounds",
            },
        )
    if grid_map.is_occupied_index(start[0], start[1]) or grid_map.is_occupied_index(goal[0], goal[1]):
        dt = float(time.perf_counter() - t0)
        return PlannerResult(
            path_xy_cells=[(float(start[0]), float(start[1]))],
            path_theta_rad=None,
            time_s=dt,
            success=False,
            stats={
                "expanded_nodes": 0,
                "path_cost": float("inf"),
                "failure_reason": "start_or_goal_in_collision",
            },
        )

    if start == goal:
        dt = float(time.perf_counter() - t0)
        return PlannerResult(
            path_xy_cells=[(float(start[0]), float(start[1]))],
            path_theta_rad=None,
            time_s=dt,
            success=True,
            stats={
                "expanded_nodes": 0,
                "path_cost": 0.0,
                "failure_reason": "",
            },
        )

    def h_cost(node: tuple[int, int]) -> float:
        dx = float(goal[0] - node[0])
        dy = float(goal[1] - node[1])
        return math.hypot(dx, dy)

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    push_id = 0
    heappush(open_heap, (h_cost(start), push_id, start))

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()
    expanded_nodes = 0
    failure_reason = "no_path"

    while open_heap:
        if float(time.perf_counter() - t0) > float(timeout_s):
            failure_reason = "timeout"
            break
        if expanded_nodes >= int(max_expanded):
            failure_reason = "max_expanded"
            break

        _f, _pid, cur = heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded_nodes += 1

        if cur == goal:
            path_cells: list[tuple[int, int]] = [cur]
            while path_cells[-1] in parent:
                path_cells.append(parent[path_cells[-1]])
            path_cells.reverse()
            dt = float(time.perf_counter() - t0)
            path = [(float(x), float(y)) for x, y in path_cells]
            return PlannerResult(
                path_xy_cells=path,
                path_theta_rad=None,
                time_s=dt,
                success=True,
                stats={
                    "expanded_nodes": int(expanded_nodes),
                    "path_cost": float(g_score.get(cur, 0.0) * cell_size_m),
                    "failure_reason": "",
                },
            )

        cur_g = float(g_score[cur])
        for dx, dy, step in neighbors:
            nx = int(cur[0] + dx)
            ny = int(cur[1] + dy)
            nxt = (nx, ny)
            if nxt in closed:
                continue
            if not grid_map.in_bounds(nx, ny):
                continue
            if grid_map.is_occupied_index(nx, ny):
                continue
            if dx != 0 and dy != 0:
                if grid_map.is_occupied_index(cur[0] + dx, cur[1]) or grid_map.is_occupied_index(cur[0], cur[1] + dy):
                    continue

            tentative = cur_g + float(step)
            prev = g_score.get(nxt)
            if prev is None or tentative < float(prev):
                g_score[nxt] = float(tentative)
                parent[nxt] = cur
                push_id += 1
                f_score = float(tentative + h_cost(nxt))
                heappush(open_heap, (f_score, push_id, nxt))

    dt = float(time.perf_counter() - t0)
    return PlannerResult(
        path_xy_cells=[(float(start[0]), float(start[1]))],
        path_theta_rad=None,
        time_s=dt,
        success=False,
        stats={
            "expanded_nodes": int(expanded_nodes),
            "path_cost": float("inf"),
            "failure_reason": str(failure_reason),
        },
    )


def plan_grid_astar(
    *,
    grid_map: GridMap,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    timeout_s: float = 5.0,
    max_expanded: int = 1_000_000,
    seed: int = 0,
    replan_attempt: int = 0,
    curve_opt_cfg: AStarCurveOptConfig | None = None,
    footprint: OrientedBoxFootprint | TwoCircleFootprint | None = None,
    ackermann_params: AckermannParams | None = None,
) -> PlannerResult:
    base_neighbors: tuple[tuple[int, int, float], ...] = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    )

    cfg = curve_opt_cfg
    use_curve_opt = bool(cfg is not None and bool(cfg.enable))
    attempts = max(1, int(cfg.max_replans) if cfg is not None else 1)
    if use_curve_opt:
        if footprint is None or ackermann_params is None:
            raise ValueError("curve optimization requires footprint and ackermann_params")

    total_time_s = 0.0
    total_expanded = 0
    last_failure = "no_path"

    inflation_base_m = 0.0
    inflation_factors: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
    if use_curve_opt:
        inflation_base_m = _curve_opt_search_inflation_margin_m(
            footprint=footprint,
            params=ackermann_params,
        )

    for local_try in range(int(attempts)):
        local_seed = int(seed) + 104729 * int(replan_attempt) + 1009 * int(local_try)
        rng = np.random.default_rng(local_seed)
        neigh = list(base_neighbors)
        rng.shuffle(neigh)

        search_map = grid_map
        search_inflation_margin_m = 0.0
        if use_curve_opt and float(inflation_base_m) > 1e-9:
            fac = float(inflation_factors[int(local_try) % int(len(inflation_factors))])
            margin_m = max(0.0, float(inflation_base_m) * fac)
            if margin_m > 1e-9:
                inflated_map = grid_map.inflate(margin_m)
                sx, sy = int(start_xy[0]), int(start_xy[1])
                gx, gy = int(goal_xy[0]), int(goal_xy[1])
                if not inflated_map.is_occupied_index(sx, sy) and not inflated_map.is_occupied_index(gx, gy):
                    search_map = inflated_map
                    search_inflation_margin_m = float(margin_m)

        res = _plan_grid_astar_once(
            grid_map=search_map,
            start_xy=start_xy,
            goal_xy=goal_xy,
            timeout_s=float(timeout_s),
            max_expanded=int(max_expanded),
            neighbors=tuple(neigh),
        )

        total_time_s += float(res.time_s)
        total_expanded += int(res.stats.get("expanded_nodes", 0))

        if not bool(res.success):
            last_failure = str(res.stats.get("failure_reason", "planner_fail"))
            continue

        if not use_curve_opt:
            s = dict(res.stats)
            s["attempt"] = int(local_try)
            s["seed"] = int(local_seed)
            s["total_replans"] = int(local_try + 1)
            s["total_time_s"] = float(total_time_s)
            s["total_expanded_nodes"] = int(total_expanded)
            s["search_inflation_margin_m"] = float(search_inflation_margin_m)
            return PlannerResult(
                path_xy_cells=list(res.path_xy_cells),
                path_theta_rad=None,
                time_s=float(total_time_s),
                success=True,
                stats=s,
            )

        if int(len(res.path_xy_cells)) <= 1:
            s = dict(res.stats)
            s["attempt"] = int(local_try)
            s["seed"] = int(local_seed)
            s["curve_opt"] = True
            s["curve_opt_trivial"] = True
            s["search_inflation_margin_m"] = float(search_inflation_margin_m)
            s["total_replans"] = int(local_try + 1)
            s["total_time_s"] = float(total_time_s)
            s["total_expanded_nodes"] = int(total_expanded)
            return PlannerResult(
                path_xy_cells=list(res.path_xy_cells),
                path_theta_rad=[0.0],
                time_s=float(total_time_s),
                success=True,
                stats=s,
            )

        opt = _optimize_grid_astar_path(
            grid_map=grid_map,
            footprint=footprint,
            params=ackermann_params,
            path_xy_cells=list(res.path_xy_cells),
            cfg=cfg,
        )
        if opt is None:
            last_failure = "curve_opt_failed"
            continue

        path_xy_cells, path_theta_rad = opt
        s = dict(res.stats)
        s["attempt"] = int(local_try)
        s["seed"] = int(local_seed)
        s["curve_opt"] = True
        s["curve_opt_resample_ds_m"] = float(cfg.resample_ds_m)
        s["curve_opt_collision_step_m"] = float(cfg.collision_step_m)
        s["curve_opt_shortcut_passes"] = int(cfg.shortcut_passes)
        s["search_inflation_margin_m"] = float(search_inflation_margin_m)
        s["total_replans"] = int(local_try + 1)
        s["total_time_s"] = float(total_time_s)
        s["total_expanded_nodes"] = int(total_expanded)
        return PlannerResult(
            path_xy_cells=list(path_xy_cells),
            path_theta_rad=list(path_theta_rad),
            time_s=float(total_time_s),
            success=True,
            stats=s,
        )

    return PlannerResult(
        path_xy_cells=[(float(start_xy[0]), float(start_xy[1]))],
        path_theta_rad=None,
        time_s=float(total_time_s),
        success=False,
        stats={
            "expanded_nodes": int(total_expanded),
            "path_cost": float("inf"),
            "failure_reason": str(last_failure),
            "total_replans": int(attempts),
            "total_time_s": float(total_time_s),
        },
    )


def default_ackermann_params(
    *,
    wheelbase_m: float = 0.6,
    delta_max_rad: float = math.radians(27.0),
    v_max_m_s: float = 2.0,
) -> AckermannParams:
    min_turn_radius_m = float(wheelbase_m) / max(1e-9, float(math.tan(float(delta_max_rad))))
    return AckermannParams(
        wheelbase=float(wheelbase_m),
        min_turn_radius=float(min_turn_radius_m),
        v_max=float(v_max_m_s),
    )


def grid_map_from_obstacles(*, grid_y0_bottom: np.ndarray, cell_size_m: float) -> GridMap:
    g = np.asarray(grid_y0_bottom, dtype=np.uint8)
    if g.ndim != 2:
        raise ValueError("grid_y0_bottom must be a (H,W) array")
    if not (float(cell_size_m) > 0.0):
        raise ValueError("cell_size_m must be > 0")
    return GridMap(g, resolution=float(cell_size_m), origin=(0.0, 0.0))


def point_footprint(*, cell_size_m: float) -> OrientedBoxFootprint:
    # Keep a small non-zero size so the unified collision checker does not
    # degenerate on strict comparisons.
    s = max(1e-3, 0.1 * float(cell_size_m))
    return OrientedBoxFootprint(length=s, width=s)


def forest_oriented_box_footprint() -> OrientedBoxFootprint:
    # Matches the forest env's nominal vehicle dimensions used by the two-circle approximation:
    # length=0.924m, width=0.740m.
    return OrientedBoxFootprint(length=0.924, width=0.740)


def forest_two_circle_footprint(*, wheelbase_m: float = 0.6) -> TwoCircleFootprint:
    # Use the same nominal vehicle dimensions as the forest env and convert to a conservative
    # two-circle approximation (robust for grid collision checks at arbitrary headings).
    #
    # IMPORTANT: The forest env's bicycle model state is the rear-axle center. To match that
    # reference, shift the footprint forward by wheelbase/2 so the two-circle model covers
    # the vehicle body centered around the axle midpoint.
    box = forest_oriented_box_footprint()
    return TwoCircleFootprint.from_box(
        length=float(box.length),
        width=float(box.width),
        center_shift=0.5 * float(wheelbase_m),
    )


def _default_start_theta(start_xy: tuple[int, int], goal_xy: tuple[int, int], *, cell_size_m: float) -> float:
    dx = float(goal_xy[0] - start_xy[0]) * float(cell_size_m)
    dy = float(goal_xy[1] - start_xy[1]) * float(cell_size_m)
    return float(math.atan2(dy, dx))


def plan_hybrid_astar(
    *,
    grid_map: GridMap,
    footprint: OrientedBoxFootprint | TwoCircleFootprint,
    params: AckermannParams,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    goal_theta_rad: float | None = 0.0,
    start_theta_rad: float | None = None,
    goal_xy_tol_m: float = 0.1,
    goal_theta_tol_rad: float = math.pi,
    timeout_s: float = 5.0,
    max_nodes: int = 200_000,
    collision_padding_m: float | None = None,
) -> PlannerResult:
    cell_size_m = float(grid_map.resolution)

    collision_padding = None
    if collision_padding_m is not None:
        collision_padding = max(0.0, float(collision_padding_m))
    planner = HybridAStarPlanner(
        grid_map,
        footprint,
        params,
        goal_xy_tol=float(goal_xy_tol_m),
        goal_theta_tol=float(goal_theta_tol_rad),
        collision_padding=collision_padding,
    )

    st = float(start_theta_rad) if start_theta_rad is not None else _default_start_theta(start_xy, goal_xy, cell_size_m=cell_size_m)
    start = AckermannState(float(start_xy[0]) * cell_size_m, float(start_xy[1]) * cell_size_m, st)

    gx_m = float(goal_xy[0]) * cell_size_m
    gy_m = float(goal_xy[1]) * cell_size_m

    gt = float(goal_theta_rad) if goal_theta_rad is not None else float(st)
    if goal_theta_rad is None:
        # Forest-style evaluation cares about reaching a goal region, not a specific heading.
        # Pick a collision-free goal heading to avoid spurious "goal_in_collision" failures
        # when the rear-axle cell is valid but the shifted footprint collides at an arbitrary yaw.
        def wrap_pi(x: float) -> float:
            return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)

        candidates = [wrap_pi(float(st) + (math.pi / 4.0) * float(k)) for k in range(8)]
        for th in candidates:
            if not bool(planner.collision_checker.collides_pose(gx_m, gy_m, float(th))):
                gt = float(th)
                break

    goal = AckermannState(gx_m, gy_m, float(gt))

    t0 = time.perf_counter()
    path, stats = planner.plan(start, goal, timeout=float(timeout_s), max_nodes=int(max_nodes), self_check=True)
    t1 = time.perf_counter()
    dt = float(stats.get("time", t1 - t0))

    if path:
        pts = [(float(s.x) / cell_size_m, float(s.y) / cell_size_m) for s in path]
        thetas = [float(s.theta) for s in path]
        return PlannerResult(path_xy_cells=pts, path_theta_rad=thetas, time_s=dt, success=True, stats=stats)
    return PlannerResult(
        path_xy_cells=[(float(start_xy[0]), float(start_xy[1]))],
        path_theta_rad=None,
        time_s=dt,
        success=False,
        stats=stats,
    )


def plan_rrt_star(
    *,
    grid_map: GridMap,
    footprint: OrientedBoxFootprint | TwoCircleFootprint,
    params: AckermannParams,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    seed: int = 0,
    goal_theta_rad: float | None = 0.0,
    start_theta_rad: float | None = None,
    goal_xy_tol_m: float = 0.1,
    goal_theta_tol_rad: float = math.pi,
    timeout_s: float = 5.0,
    max_iter: int = 5_000,
    collision_padding_m: float | None = None,
) -> PlannerResult:
    cell_size_m = float(grid_map.resolution)
    st = float(start_theta_rad) if start_theta_rad is not None else _default_start_theta(start_xy, goal_xy, cell_size_m=cell_size_m)
    start = AckermannState(float(start_xy[0]) * cell_size_m, float(start_xy[1]) * cell_size_m, st)

    collision_padding = None
    if collision_padding_m is not None:
        collision_padding = max(0.0, float(collision_padding_m))
    planner = RRTStarPlanner(
        grid_map,
        footprint,
        params,
        rng_seed=int(seed),
        goal_xy_tol=float(goal_xy_tol_m),
        goal_theta_tol=float(goal_theta_tol_rad),
        collision_padding=collision_padding,
    )

    gx_m = float(goal_xy[0]) * cell_size_m
    gy_m = float(goal_xy[1]) * cell_size_m

    gt = float(goal_theta_rad) if goal_theta_rad is not None else float(st)
    if goal_theta_rad is None:
        def wrap_pi(x: float) -> float:
            return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)

        candidates = [wrap_pi(float(st) + (math.pi / 4.0) * float(k)) for k in range(8)]
        for th in candidates:
            if not bool(planner.collision_checker.collides_pose(gx_m, gy_m, float(th))):
                gt = float(th)
                break

    goal = AckermannState(gx_m, gy_m, float(gt))

    t0 = time.perf_counter()
    path, stats = planner.plan(
        start,
        goal,
        timeout=float(timeout_s),
        max_iter=int(max_iter),
        self_check=False,
    )
    t1 = time.perf_counter()
    dt = float(stats.get("time", t1 - t0))

    success = bool(stats.get("success", bool(path)))
    if path:
        pts = [(float(s.x) / cell_size_m, float(s.y) / cell_size_m) for s in path]
        thetas = [float(s.theta) for s in path]
        return PlannerResult(path_xy_cells=pts, path_theta_rad=thetas, time_s=dt, success=success, stats=stats)
    return PlannerResult(
        path_xy_cells=[(float(start_xy[0]), float(start_xy[1]))],
        path_theta_rad=None,
        time_s=dt,
        success=False,
        stats=stats,
    )
