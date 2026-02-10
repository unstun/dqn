from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from forest_vehicle_dqn.env import AMRBicycleEnv


@dataclass(frozen=True)
class MPCConfig:
    horizon_steps: int = 12
    candidates: int = 256
    dt_s: float = 0.0
    w_u: float = 0.05
    w_du: float = 0.5
    w_pos: float = 6.0
    w_yaw: float = 1.2
    align_dist_m: float = 1.5
    collision_padding_m: float = 0.0
    goal_lookahead_steps: int = 2


@dataclass(frozen=True)
class MPCPlanResult:
    path_xy_cells: list[tuple[float, float]]
    path_xy_m: list[tuple[float, float]]
    compute_time_s: float
    reached: bool
    collision: bool
    truncated: bool
    steps: int
    path_time_s: float
    failure_reason: str
    t_s: np.ndarray
    v_m_s: np.ndarray
    delta_rad: np.ndarray
    debug: dict[str, object]


@dataclass(frozen=True)
class MPCStepResult:
    a_m_s2: float
    delta_dot_rad_s: float
    feasible: bool
    best_cost: float
    nearest_path_idx: int
    fail_reason: str


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (np.remainder(np.asarray(x, dtype=np.float64) + math.pi, 2.0 * math.pi) - math.pi).astype(np.float64, copy=False)


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


def _nearest_path_index(*, x_m: float, y_m: float, path_xy_m: np.ndarray, prev_idx: int) -> int:
    n = int(path_xy_m.shape[0])
    if n <= 1:
        return 0
    lo = max(0, int(prev_idx) - 25)
    hi = min(n, int(prev_idx) + 220)
    if hi <= lo:
        lo, hi = 0, n
    chunk = path_xy_m[lo:hi]
    dx = chunk[:, 0] - float(x_m)
    dy = chunk[:, 1] - float(y_m)
    idx_rel = int(np.argmin(dx * dx + dy * dy))
    return int(lo + idx_rel)


def _control_grid(*, candidates: int, a_max: float, delta_dot_max: float) -> tuple[np.ndarray, np.ndarray]:
    c = max(9, int(candidates))
    n_a = max(3, int(math.sqrt(c)))
    n_dd = max(3, int(math.ceil(c / float(n_a))))
    a_vals = np.linspace(-float(a_max), float(a_max), num=n_a, dtype=np.float64)
    dd_vals = np.linspace(-float(delta_dot_max), float(delta_dot_max), num=n_dd, dtype=np.float64)
    aa, dd = np.meshgrid(a_vals, dd_vals, indexing="ij")
    a_cand = aa.reshape(-1)
    dd_cand = dd.reshape(-1)
    if a_cand.size > int(c):
        a_cand = a_cand[: int(c)]
        dd_cand = dd_cand[: int(c)]
    return a_cand.astype(np.float64, copy=False), dd_cand.astype(np.float64, copy=False)


def solve_mpc_one_step(
    *,
    env: AMRBicycleEnv,
    path_xy_m: np.ndarray,
    path_theta: np.ndarray,
    prev_path_idx: int,
    config: MPCConfig,
) -> MPCStepResult:
    if path_xy_m.ndim != 2 or int(path_xy_m.shape[0]) < 2:
        return MPCStepResult(
            a_m_s2=0.0,
            delta_dot_rad_s=0.0,
            feasible=False,
            best_cost=float("inf"),
            nearest_path_idx=0,
            fail_reason="planner_path_too_short",
        )
    if path_theta.ndim != 1 or int(path_theta.size) != int(path_xy_m.shape[0]):
        return MPCStepResult(
            a_m_s2=0.0,
            delta_dot_rad_s=0.0,
            feasible=False,
            best_cost=float("inf"),
            nearest_path_idx=0,
            fail_reason="invalid_path_theta",
        )

    dt = float(config.dt_s) if float(config.dt_s) > 0.0 else float(env.model.dt)
    dt = max(1e-6, float(dt))

    horizon = max(1, int(config.horizon_steps))
    lookahead = max(1, int(config.goal_lookahead_steps))
    n_cand_req = max(9, int(config.candidates))

    a_max = abs(float(env.model.a_max_m_s2))
    delta_dot_max = abs(float(env.model.delta_dot_max_rad_s))
    delta_max = abs(float(env.model.delta_max_rad))
    v_max = abs(float(env.model.v_max_m_s))
    wheelbase = max(1e-6, float(env.model.wheelbase_m))

    w_u = float(config.w_u)
    w_du = float(config.w_du)
    w_pos = float(config.w_pos)
    w_yaw = float(config.w_yaw)
    align_dist = max(0.0, float(config.align_dist_m))
    eps = max(0.0, float(config.collision_padding_m))

    a_cand, dd_cand = _control_grid(
        candidates=int(n_cand_req),
        a_max=float(a_max),
        delta_dot_max=float(delta_dot_max),
    )
    n_cand = int(a_cand.size)

    x0 = float(env._x_m)
    y0 = float(env._y_m)
    psi0 = float(env._psi_rad)
    v0 = float(env._v_m_s)
    delta0 = float(env._delta_rad)

    gx_m = float(env.goal_xy[0]) * float(env.cell_size_m)
    gy_m = float(env.goal_xy[1]) * float(env.cell_size_m)
    psi_d = float(path_theta[-1]) if int(path_theta.size) > 0 else float(math.atan2(float(gy_m - y0), float(gx_m - x0)))
    if not math.isfinite(float(psi_d)):
        psi_d = float(math.atan2(float(gy_m - y0), float(gx_m - x0)))

    nearest_idx = _nearest_path_index(x_m=x0, y_m=y0, path_xy_m=path_xy_m, prev_idx=int(prev_path_idx))
    ref_idx = np.minimum(
        int(nearest_idx) + (np.arange(horizon, dtype=np.int64) + 1) * int(lookahead),
        int(path_xy_m.shape[0]) - 1,
    )
    ref_x = path_xy_m[ref_idx, 0]
    ref_y = path_xy_m[ref_idx, 1]

    x = np.full((n_cand,), float(x0), dtype=np.float64)
    y = np.full((n_cand,), float(y0), dtype=np.float64)
    psi = np.full((n_cand,), float(psi0), dtype=np.float64)
    v = np.full((n_cand,), float(v0), dtype=np.float64)
    delta = np.full((n_cand,), float(delta0), dtype=np.float64)

    prev_a0 = float(getattr(env, "_prev_a", 0.0))
    prev_a_cmd = np.full((n_cand,), float(prev_a0), dtype=np.float64)
    prev_delta_cmd = np.full((n_cand,), float(delta0), dtype=np.float64)

    costs = np.zeros((n_cand,), dtype=np.float64)
    alive = np.ones((n_cand,), dtype=np.bool_)

    for k in range(horizon):
        if not bool(alive.any()):
            break

        v = np.clip(v + a_cand * dt, -float(v_max), float(v_max))
        delta = np.clip(delta + dd_cand * dt, -float(delta_max), float(delta_max))

        x = x + v * np.cos(psi) * dt
        y = y + v * np.sin(psi) * dt
        psi = _wrap_angle_np(psi + (v / max(1e-9, float(wheelbase))) * np.tan(delta) * dt)

        od, coll = env._od_and_collision_at_pose_m_vec(x, y, psi)
        if eps > 0.0:
            coll = np.asarray(coll, dtype=np.bool_) | (np.asarray(od, dtype=np.float64) < float(eps))

        alive = alive & (~np.asarray(coll, dtype=np.bool_))
        if not bool(alive.any()):
            break

        a_cmd = a_cand
        du_a = a_cmd - prev_a_cmd
        du_delta = _wrap_angle_np(delta - prev_delta_cmd)

        pos_err2 = np.square(x - float(ref_x[k])) + np.square(y - float(ref_y[k]))
        goal_dist = np.hypot(float(gx_m) - x, float(gy_m) - y)
        near_goal = (goal_dist < float(align_dist)).astype(np.float64, copy=False)
        yaw_err = _wrap_angle_np(psi - float(psi_d))

        stage = (
            float(w_u) * (np.square(a_cmd) + np.square(delta))
            + float(w_du) * (np.square(du_a) + np.square(du_delta))
            + float(w_pos) * pos_err2
            + near_goal * float(w_yaw) * np.square(yaw_err)
        )
        costs = costs + np.where(alive, stage, 0.0)

        prev_a_cmd = a_cmd
        prev_delta_cmd = delta

    costs = np.where(alive, costs, float("inf"))
    if not np.isfinite(costs).any():
        return MPCStepResult(
            a_m_s2=0.0,
            delta_dot_rad_s=0.0,
            feasible=False,
            best_cost=float("inf"),
            nearest_path_idx=int(nearest_idx),
            fail_reason="all_candidates_infeasible",
        )

    best = int(np.argmin(costs))
    a_best = float(np.clip(a_cand[best], -float(a_max), float(a_max)))
    dd_best = float(np.clip(dd_cand[best], -float(delta_dot_max), float(delta_dot_max)))

    return MPCStepResult(
        a_m_s2=float(a_best),
        delta_dot_rad_s=float(dd_best),
        feasible=True,
        best_cost=float(costs[best]),
        nearest_path_idx=int(nearest_idx),
        fail_reason="",
    )


def run_mpc_local_planning(
    *,
    env: AMRBicycleEnv,
    plan_path_xy_cells: list[tuple[float, float]],
    plan_path_theta_rad: list[float] | None,
    max_steps: int,
    seed: int,
    config: MPCConfig,
    reset_options: dict[str, object] | None = None,
) -> MPCPlanResult:
    if len(plan_path_xy_cells) < 2:
        return MPCPlanResult(
            path_xy_cells=[(float(env.start_xy[0]), float(env.start_xy[1]))],
            path_xy_m=[(float(env.start_xy[0]) * float(env.cell_size_m), float(env.start_xy[1]) * float(env.cell_size_m))],
            compute_time_s=0.0,
            reached=False,
            collision=False,
            truncated=False,
            steps=0,
            path_time_s=0.0,
            failure_reason="planner_path_too_short",
            t_s=np.zeros((1,), dtype=np.float64),
            v_m_s=np.zeros((1,), dtype=np.float64),
            delta_rad=np.zeros((1,), dtype=np.float64),
            debug={"mpc_steps": 0},
        )

    env.reset(seed=int(seed), options=reset_options)

    cell_size_m = float(env.cell_size_m)
    path_xy_m = np.asarray(
        [(float(x) * cell_size_m, float(y) * cell_size_m) for x, y in plan_path_xy_cells],
        dtype=np.float64,
    )
    if path_xy_m.ndim != 2 or path_xy_m.shape[0] < 2:
        raise ValueError("plan_path_xy_cells must provide at least two points")

    if plan_path_theta_rad is not None and len(plan_path_theta_rad) == int(path_xy_m.shape[0]):
        path_theta = np.asarray([float(t) for t in plan_path_theta_rad], dtype=np.float64)
    else:
        path_theta = _path_headings(path_xy_m)

    dt = float(config.dt_s) if float(config.dt_s) > 0.0 else float(env.model.dt)

    t_hist: list[float] = [0.0]
    v_hist: list[float] = [float(getattr(env, "_v_m_s", 0.0))]
    delta_hist: list[float] = [float(getattr(env, "_delta_rad", 0.0))]

    path_cells_exec: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]
    path_m_exec: list[tuple[float, float]] = [(float(getattr(env, "_x_m", 0.0)), float(getattr(env, "_y_m", 0.0)))]

    prev_path_idx = 0
    mpc_steps = 0
    done = False
    truncated = False
    reached = False
    collision = False
    fail_reason = "max_steps"

    t0 = time.perf_counter()

    while (not done) and (not truncated) and mpc_steps < int(max_steps):
        mpc_steps += 1

        step = solve_mpc_one_step(
            env=env,
            path_xy_m=path_xy_m,
            path_theta=path_theta,
            prev_path_idx=int(prev_path_idx),
            config=config,
        )
        prev_path_idx = int(step.nearest_path_idx)

        if not bool(step.feasible):
            a_best = 0.0
            dd_best = 0.0
            fail_reason = str(step.fail_reason)
        else:
            a_best = float(step.a_m_s2)
            dd_best = float(step.delta_dot_rad_s)

        _obs, _reward, done, truncated, info = env.step_continuous(delta_dot_rad_s=float(dd_best), a_m_s2=float(a_best))

        pxy = info.get("agent_xy", (float(env._x_m) / cell_size_m, float(env._y_m) / cell_size_m))
        path_cells_exec.append((float(pxy[0]), float(pxy[1])))
        pose = info.get("pose_m", (float(env._x_m), float(env._y_m), float(env._psi_rad)))
        path_m_exec.append((float(pose[0]), float(pose[1])))

        t_hist.append(float(mpc_steps) * float(dt))
        v_hist.append(float(info.get("v_m_s", float(env._v_m_s))))
        delta_hist.append(float(info.get("delta_rad", float(env._delta_rad))))

        reached = bool(info.get("reached", False))
        collision = bool(info.get("collision", False) or info.get("stuck", False))

        if reached:
            fail_reason = "reached"
            break
        if collision:
            fail_reason = "collision"
            break
        if bool(truncated):
            fail_reason = "truncated"
            break

    compute_t = float(time.perf_counter() - t0)
    path_time_s = float(mpc_steps) * float(dt)

    return MPCPlanResult(
        path_xy_cells=list(path_cells_exec),
        path_xy_m=list(path_m_exec),
        compute_time_s=float(compute_t),
        reached=bool(reached),
        collision=bool(collision),
        truncated=bool(truncated),
        steps=int(mpc_steps),
        path_time_s=float(path_time_s),
        failure_reason=str(fail_reason),
        t_s=np.asarray(t_hist, dtype=np.float64),
        v_m_s=np.asarray(v_hist, dtype=np.float64),
        delta_rad=np.asarray(delta_hist, dtype=np.float64),
        debug={
            "horizon_steps": int(max(1, int(config.horizon_steps))),
            "candidates": int(max(9, int(config.candidates))),
            "dt_s": float(dt),
            "goal_lookahead_steps": int(max(1, int(config.goal_lookahead_steps))),
            "collision_padding_m": float(max(0.0, float(config.collision_padding_m))),
            "align_dist_m": float(max(0.0, float(config.align_dist_m))),
            "weights": {
                "w_u": float(config.w_u),
                "w_du": float(config.w_du),
                "w_pos": float(config.w_pos),
                "w_yaw": float(config.w_yaw),
            },
        },
    )
