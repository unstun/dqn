from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from forest_vehicle_dqn.agents import AgentConfig, DQNFamilyAgent
from forest_vehicle_dqn.baselines.mpc_local_planner import MPCConfig, solve_mpc_one_step
from forest_vehicle_dqn.baselines.pathplan import (
    AStarCurveOptConfig,
    PlannerResult,
    default_ackermann_params,
    forest_two_circle_footprint,
    grid_map_from_obstacles,
    plan_grid_astar,
    plan_hybrid_astar,
    plan_rrt_star,
)
from forest_vehicle_dqn.config_io import apply_config_defaults, load_json, resolve_config_path, select_section
from forest_vehicle_dqn.env import AMRBicycleEnv, wrap_angle_rad
from forest_vehicle_dqn.maps import FOREST_ENV_ORDER, get_map_spec
from forest_vehicle_dqn.runtime import configure_runtime, select_device

configure_runtime()

PlannerName = Literal["hybrid_astar", "rrt_star", "grid_astar", "cnn_ddqn"]


@dataclass(frozen=True)
class GamePlan:
    planner: str
    path_xy_cells: list[tuple[float, float]]
    path_theta_rad: list[float] | None
    time_s: float
    success: bool
    stats: dict[str, Any]


@dataclass
class MPCTracker:
    cfg: MPCConfig
    prev_path_idx: int = 0

    def reset(self) -> None:
        self.prev_path_idx = 0


@dataclass
class _PlanJob:
    request_id: int
    planner: PlannerName
    start_xy: tuple[int, int]
    goal_xy: tuple[int, int]
    start_theta_rad: float
    seed: int


def _path_headings(path_xy_m: np.ndarray) -> np.ndarray:
    pts = np.asarray(path_xy_m, dtype=np.float64)
    n = int(pts.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    dxy = np.diff(pts, axis=0)
    seg_h = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float64, copy=False)
    out = np.zeros((n,), dtype=np.float64)
    out[:-1] = seg_h
    out[-1] = seg_h[-1]
    return out


def _snap_to_free_cell(
    grid_y0_bottom: np.ndarray, *, xy: tuple[int, int], max_radius: int = 10
) -> tuple[int, int] | None:
    gx, gy = int(xy[0]), int(xy[1])
    h, w = grid_y0_bottom.shape
    if not (0 <= gx < w and 0 <= gy < h):
        return None
    if not bool(grid_y0_bottom[gy, gx]):
        return (gx, gy)

    r_max = int(max(0, int(max_radius)))
    for r in range(1, r_max + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                x = gx + dx
                y = gy + dy
                if 0 <= x < w and 0 <= y < h and not bool(grid_y0_bottom[y, x]):
                    return (int(x), int(y))
    return None


def _build_env(*, env_name: str, max_steps: int, seed: int) -> tuple[AMRBicycleEnv, np.ndarray]:
    spec = get_map_spec(str(env_name))
    grid = spec.obstacle_grid()
    env = AMRBicycleEnv(spec, max_steps=int(max_steps))
    env.reset(seed=int(seed))
    return env, grid


def _planner_from_cli(value: str) -> PlannerName:
    v = str(value).strip().lower().replace("-", "_")
    if v in {"hybrid_astar", "rrt_star", "grid_astar", "cnn_ddqn"}:
        return v  # type: ignore[return-value]
    raise ValueError("planner must be one of: hybrid_astar rrt_star grid_astar cnn_ddqn")


def _rollout_cnn_ddqn_path(
    *,
    env_name: str,
    grid_y0_bottom: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    start_theta_rad: float,
    checkpoint: Path,
    device: str,
    max_steps: int,
    seed: int,
) -> PlannerResult:
    spec = get_map_spec(str(env_name))
    env = AMRBicycleEnv(spec, max_steps=int(max_steps))
    obs, _info = env.reset(seed=int(seed), options={"start_xy": start_xy, "goal_xy": goal_xy})
    env._psi_rad = float(wrap_angle_rad(float(start_theta_rad)))

    agent = DQNFamilyAgent(
        algo="cnn-ddqn",
        obs_dim=int(env.observation_space.shape[0]),
        n_actions=int(env.action_space.n),
        config=AgentConfig(),
        seed=int(seed),
        device=str(device),
    )
    agent.load(Path(checkpoint))

    path_xy_cells: list[tuple[float, float]] = [env._agent_xy_for_plot()]
    reached = False
    collision = False
    truncated = False

    t0 = time.perf_counter()
    for _ in range(int(max_steps)):
        a = int(agent.act(np.asarray(obs, dtype=np.float32), episode=0, explore=False))
        obs, _reward, terminated, truncated, info = env.step(a)
        pxy = info.get("agent_xy", env._agent_xy_for_plot())
        path_xy_cells.append((float(pxy[0]), float(pxy[1])))
        reached = bool(info.get("reached", False))
        collision = bool(info.get("collision", False) or info.get("stuck", False))
        if reached or collision or bool(terminated) or bool(truncated):
            break
    t1 = time.perf_counter()

    return PlannerResult(
        path_xy_cells=list(path_xy_cells),
        path_theta_rad=None,
        time_s=float(t1 - t0),
        success=bool(reached),
        stats={
            "planner": "cnn_ddqn",
            "reached": bool(reached),
            "collision": bool(collision),
            "truncated": bool(truncated),
            "steps": int(len(path_xy_cells) - 1),
            "checkpoint": str(checkpoint),
        },
    )


def _compute_plan(
    *,
    job: _PlanJob,
    env_name: str,
    grid_y0_bottom: np.ndarray,
    cell_size_m: float,
    planner_timeout_s: float,
    hybrid_max_nodes: int,
    rrt_max_iter: int,
    astar_max_expanded: int,
    astar_curve_opt: bool,
    collision_padding_m: float,
    rl_checkpoint: Path | None,
    rl_device: str,
    rl_max_steps: int,
) -> GamePlan:
    grid_map = grid_map_from_obstacles(grid_y0_bottom=grid_y0_bottom, cell_size_m=float(cell_size_m))
    ackermann = default_ackermann_params(
        wheelbase_m=float(0.6),
        delta_max_rad=float(math.radians(27.0)),
        v_max_m_s=float(2.0),
    )

    footprint = forest_two_circle_footprint(wheelbase_m=float(ackermann.wheelbase))
    timeout_s = max(0.01, float(planner_timeout_s))

    if job.planner == "hybrid_astar":
        res = plan_hybrid_astar(
            grid_map=grid_map,
            footprint=footprint,
            params=ackermann,
            start_xy=job.start_xy,
            goal_xy=job.goal_xy,
            start_theta_rad=float(job.start_theta_rad),
            goal_theta_rad=None,
            timeout_s=float(timeout_s),
            max_nodes=int(hybrid_max_nodes),
            collision_padding_m=float(max(0.0, float(collision_padding_m))),
            goal_xy_tol_m=0.5,
            goal_theta_tol_rad=float(math.pi),
        )
        return GamePlan(
            planner="hybrid_astar",
            path_xy_cells=list(res.path_xy_cells),
            path_theta_rad=list(res.path_theta_rad) if res.path_theta_rad is not None else None,
            time_s=float(res.time_s),
            success=bool(res.success),
            stats=dict(res.stats),
        )

    if job.planner == "rrt_star":
        res = plan_rrt_star(
            grid_map=grid_map,
            footprint=footprint,
            params=ackermann,
            start_xy=job.start_xy,
            goal_xy=job.goal_xy,
            start_theta_rad=float(job.start_theta_rad),
            goal_theta_rad=None,
            timeout_s=float(timeout_s),
            max_iter=int(rrt_max_iter),
            seed=int(job.seed),
            collision_padding_m=float(max(0.0, float(collision_padding_m))),
            goal_xy_tol_m=0.5,
            goal_theta_tol_rad=float(math.pi),
        )
        return GamePlan(
            planner="rrt_star",
            path_xy_cells=list(res.path_xy_cells),
            path_theta_rad=list(res.path_theta_rad) if res.path_theta_rad is not None else None,
            time_s=float(res.time_s),
            success=bool(res.success),
            stats=dict(res.stats),
        )

    if job.planner == "grid_astar":
        curve_cfg = None
        if bool(astar_curve_opt):
            curve_cfg = AStarCurveOptConfig(enable=True, max_replans=30, resample_ds_m=0.2, collision_step_m=0.1, shortcut_passes=2)
        res = plan_grid_astar(
            grid_map=grid_map,
            start_xy=job.start_xy,
            goal_xy=job.goal_xy,
            timeout_s=float(timeout_s),
            max_expanded=int(astar_max_expanded),
            seed=int(job.seed),
            curve_opt_cfg=curve_cfg,
            footprint=footprint if curve_cfg is not None else None,
            ackermann_params=ackermann if curve_cfg is not None else None,
        )
        return GamePlan(
            planner="grid_astar",
            path_xy_cells=list(res.path_xy_cells),
            path_theta_rad=list(res.path_theta_rad) if res.path_theta_rad is not None else None,
            time_s=float(res.time_s),
            success=bool(res.success),
            stats=dict(res.stats),
        )

    if job.planner == "cnn_ddqn":
        if rl_checkpoint is None:
            return GamePlan(
                planner="cnn_ddqn",
                path_xy_cells=[(float(job.start_xy[0]), float(job.start_xy[1]))],
                path_theta_rad=None,
                time_s=0.0,
                success=False,
                stats={"failure_reason": "missing --rl-checkpoint"},
            )
        res = _rollout_cnn_ddqn_path(
            env_name=env_name,
            grid_y0_bottom=grid_y0_bottom,
            start_xy=job.start_xy,
            goal_xy=job.goal_xy,
            start_theta_rad=float(job.start_theta_rad),
            checkpoint=Path(rl_checkpoint),
            device=str(rl_device),
            max_steps=int(rl_max_steps),
            seed=int(job.seed),
        )
        return GamePlan(
            planner="cnn_ddqn",
            path_xy_cells=list(res.path_xy_cells),
            path_theta_rad=list(res.path_theta_rad) if res.path_theta_rad is not None else None,
            time_s=float(res.time_s),
            success=bool(res.success),
            stats=dict(res.stats),
        )

    raise AssertionError(f"Unknown planner: {job.planner}")


class _PygameViewer:
    def __init__(
        self,
        *,
        window_size: int = 900,
        fps: int = 60,
        trail_len: int = 300,
        show_collision_box: bool = True,
        logger: Any | None = None,
    ) -> None:
        self.window_size = max(200, int(window_size))
        self.fps = max(0, int(fps))
        self.trail_len = max(1, int(trail_len))
        self.show_collision_box = bool(show_collision_box)
        self._log = logger if logger is not None else self._default_log

        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._clock: Any | None = None
        self._font: Any | None = None

        self._map_surface: Any | None = None
        self._map_shape: tuple[int, int] | None = None
        self._cell_px: float = 1.0
        self._map_left: int = 0
        self._map_top: int = 0
        self._map_w_px: int = 1
        self._map_h_px: int = 1

        self.running = True
        self._status_msg = ""

    def ensure_backend(self) -> bool:
        if self._pygame is None:
            try:
                import pygame  # type: ignore
            except Exception:
                self._log("[game] pygame is missing; install via `pip install -r requirements-optional.txt`")
                return False
            self._pygame = pygame

        if self._screen is not None:
            return True

        try:
            assert self._pygame is not None
            self._pygame.init()
            self._pygame.font.init()
            self._screen = self._pygame.display.set_mode((self.window_size, self.window_size))
            self._clock = self._pygame.time.Clock()
            self._font = self._load_font(size=16)
            return True
        except Exception as exc:
            self._log(f"[game] pygame init failed: {exc}")
            return False

    def _load_font(self, *, size: int) -> Any:
        assert self._pygame is not None
        sz = int(max(8, int(size)))

        # Prefer explicit font files that can render Chinese, then fall back to SysFont/default.
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
            "/usr/share/fonts/truetype/winfonts/msyhl.ttc",  # Microsoft YaHei (often installed in lab setups)
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
        for p in candidates:
            try:
                if Path(p).is_file():
                    return self._pygame.font.Font(p, sz)
            except Exception:
                continue

        for name in (
            "Noto Sans CJK SC",
            "Microsoft YaHei",
            "WenQuanYi Micro Hei",
            "SimHei",
            "PingFang SC",
            "DejaVu Sans",
        ):
            try:
                f = self._pygame.font.SysFont(name, sz)
                if f is not None:
                    return f
            except Exception:
                continue

        return self._pygame.font.Font(None, sz)

    def close(self) -> None:
        self.running = False
        if self._pygame is None:
            return
        try:
            if self._screen is not None:
                self._pygame.display.quit()
        except Exception:
            pass
        finally:
            self._screen = None
            self._clock = None
            self._font = None

    def prepare_map(self, grid_y0_bottom: np.ndarray) -> bool:
        if not self.ensure_backend():
            return False
        grid = np.asarray(grid_y0_bottom, dtype=np.uint8)
        if grid.ndim != 2:
            self._log("[game] grid must be 2D")
            return False

        shape = (int(grid.shape[0]), int(grid.shape[1]))
        if self._map_surface is not None and self._map_shape == shape:
            return True

        assert self._pygame is not None
        pad = max(8, int(round(self.window_size * 0.04)))
        usable = max(16, self.window_size - 2 * pad)
        h, w = shape
        self._cell_px = max(1.0, min(float(usable) / float(max(1, w)), float(usable) / float(max(1, h))))
        self._map_w_px = max(1, int(round(float(w) * self._cell_px)))
        self._map_h_px = max(1, int(round(float(h) * self._cell_px)))
        self._map_left = (self.window_size - self._map_w_px) // 2
        self._map_top = (self.window_size - self._map_h_px) // 2

        surf = self._pygame.Surface((self._map_w_px, self._map_h_px))
        surf.fill((245, 245, 245))
        obs = np.argwhere(grid.astype(bool, copy=False))
        if obs.size > 0:
            cell_i = max(1, int(math.ceil(self._cell_px)))
            for y, x in obs:
                px = int(round(float(x) * self._cell_px))
                py = int(round(float(h - 1 - y) * self._cell_px))
                self._pygame.draw.rect(surf, (48, 48, 48), self._pygame.Rect(px, py, cell_i, cell_i))

        self._pygame.draw.rect(surf, (130, 130, 130), self._pygame.Rect(0, 0, self._map_w_px, self._map_h_px), 1)
        self._map_surface = surf
        self._map_shape = shape
        return True

    def grid_to_px(self, xy: tuple[float, float]) -> tuple[int, int]:
        h = int(self._map_shape[0]) if self._map_shape is not None else 1
        x = float(xy[0])
        y = float(xy[1])
        px = float(self._map_left) + (x + 0.5) * float(self._cell_px)
        py = float(self._map_top) + (float(h) - (y + 0.5)) * float(self._cell_px)
        return (int(round(px)), int(round(py)))

    def px_to_grid(self, px: int, py: int) -> tuple[float, float] | None:
        if self._map_shape is None:
            return None
        h, w = int(self._map_shape[0]), int(self._map_shape[1])
        x = (float(px) - float(self._map_left)) / float(self._cell_px) - 0.5
        y = float(h) - (float(py) - float(self._map_top)) / float(self._cell_px) - 0.5
        if not (-1.0 <= x <= float(w) + 1.0 and -1.0 <= y <= float(h) + 1.0):
            return None
        return (float(x), float(y))

    def poll_events(self) -> list[Any]:
        if not self.ensure_backend():
            self.running = False
            return []
        assert self._pygame is not None
        out: list[Any] = []
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.running = False
                break
            out.append(event)
        return out

    def set_status(self, msg: str) -> None:
        self._status_msg = str(msg)

    def render(
        self,
        *,
        env_name: str,
        planner: str,
        controller: str,
        paused: bool,
        env: AMRBicycleEnv,
        start_xy: tuple[int, int],
        goal_xy: tuple[int, int],
        trail: deque[tuple[float, float]],
        plan_path_xy_cells: list[tuple[float, float]] | None,
        plan_success: bool | None,
        plan_time_s: float | None,
        info: dict[str, object] | None,
    ) -> None:
        if not self.ensure_backend():
            self.running = False
            return
        if self._screen is None or self._pygame is None:
            return
        if self._map_surface is None:
            return

        self._screen.fill((18, 20, 24))
        self._screen.blit(self._map_surface, (self._map_left, self._map_top))

        if plan_path_xy_cells is not None and len(plan_path_xy_cells) >= 2:
            pts = [self.grid_to_px(p) for p in plan_path_xy_cells]
            color = (0, 200, 255) if bool(plan_success) else (120, 120, 120)
            self._pygame.draw.lines(self._screen, color, False, pts, 2)

        if len(trail) >= 2:
            pts = [self.grid_to_px(p) for p in trail]
            self._pygame.draw.lines(self._screen, (255, 191, 0), False, pts, 2)

        start_px = self.grid_to_px((float(start_xy[0]), float(start_xy[1])))
        goal_px = self.grid_to_px((float(goal_xy[0]), float(goal_xy[1])))
        agent_xy = trail[-1] if trail else (float(start_xy[0]), float(start_xy[1]))
        agent_px = self.grid_to_px(agent_xy)

        radius = max(3, int(round(self._cell_px * 0.28)))
        self._pygame.draw.circle(self._screen, (66, 165, 245), start_px, radius + 1)
        self._pygame.draw.circle(self._screen, (239, 83, 80), goal_px, radius + 1)

        if self.show_collision_box:
            self._draw_oriented_collision_box(env=env, collision=bool((info or {}).get("collision", False)))

        agent_color = (102, 187, 106)
        if bool((info or {}).get("collision", False)):
            agent_color = (255, 112, 67)
        elif bool((info or {}).get("reached", False)):
            agent_color = (171, 71, 188)
        self._pygame.draw.circle(self._screen, agent_color, agent_px, radius + 2)

        self._draw_text(
            env_name=str(env_name),
            planner=str(planner),
            controller=str(controller),
            paused=bool(paused),
            plan_success=plan_success,
            plan_time_s=plan_time_s,
            info=info,
        )

        try:
            self._pygame.display.set_caption(f"game: {env_name}  planner={planner} controller={controller}")
        except Exception:
            pass
        self._pygame.display.flip()

        if self._clock is not None and self.fps > 0:
            self._clock.tick(self.fps)

    def _draw_text(
        self,
        *,
        env_name: str,
        planner: str,
        controller: str,
        paused: bool,
        plan_success: bool | None,
        plan_time_s: float | None,
        info: dict[str, object] | None,
    ) -> None:
        if self._screen is None or self._font is None:
            return
        lines: list[str] = [
            f"{env_name} | 规划器={planner} | 控制器={controller} | 暂停={int(paused)}",
            "鼠标左键：设置目标点  |  1/2/3/4：选择规划器",
            "R：重置  |  SPACE：暂停  |  P：重新规划",
        ]
        if plan_success is not None:
            lines.append(f"规划：成功={int(bool(plan_success))}  用时={float(plan_time_s or 0.0):.3f}s")

        if info:
            od_m = info.get("od_m", None)
            d_goal_m = info.get("d_goal_m", None)
            v_m_s = info.get("v_m_s", None)
            delta_rad = info.get("delta_rad", None)
            reached = int(bool(info.get("reached", False)))
            collision = int(bool(info.get("collision", False)))
            stuck = int(bool(info.get("stuck", False)))
            s = f"到达={reached}  碰撞={collision}  卡住={stuck}"
            if isinstance(d_goal_m, (float, int)) and math.isfinite(float(d_goal_m)):
                s += f"  距目标={float(d_goal_m):.2f}m"
            if isinstance(od_m, (float, int)) and math.isfinite(float(od_m)):
                s += f"  净空={float(od_m):.2f}m"
            if isinstance(v_m_s, (float, int)) and math.isfinite(float(v_m_s)):
                s += f"  速度={float(v_m_s):.2f}m/s"
            if isinstance(delta_rad, (float, int)) and math.isfinite(float(delta_rad)):
                s += f"  转角={math.degrees(float(delta_rad)):.1f}deg"
            lines.append(s)

        if self._status_msg:
            lines.append(str(self._status_msg))

        surfs = [self._font.render(line, True, (240, 240, 240)) for line in lines]
        if not surfs:
            return

        x0 = 10
        y0 = 8
        pad_x = 10
        pad_y = 8
        line_gap = 2

        max_w = max(int(s.get_width()) for s in surfs)
        heights = [max(18, int(s.get_height())) for s in surfs]
        content_h = int(sum(heights) + max(0, len(heights) - 1) * int(line_gap))

        panel_w = int(max_w + 2 * pad_x)
        panel_h = int(content_h + 2 * pad_y)
        panel = self._pygame.Surface((panel_w, panel_h), self._pygame.SRCALPHA)
        panel.fill((0, 0, 0, 165))
        self._screen.blit(panel, (x0, y0))

        y = int(y0 + pad_y)
        for surf, h in zip(surfs, heights):
            self._screen.blit(surf, (int(x0 + pad_x), int(y)))
            y += int(h + line_gap)

    def _draw_oriented_collision_box(self, *, env: AMRBicycleEnv, collision: bool) -> None:
        if self._pygame is None or self._screen is None:
            return
        x_m = float(getattr(env, "_x_m", 0.0))
        y_m = float(getattr(env, "_y_m", 0.0))
        psi = float(getattr(env, "_psi_rad", 0.0))
        cell_size_m = max(1e-6, float(env.cell_size_m))

        footprint = getattr(env, "footprint", None)
        if footprint is None:
            return
        try:
            x1 = float(getattr(footprint, "x1_m"))
            x2 = float(getattr(footprint, "x2_m"))
            radius = float(getattr(footprint, "radius_m"))
        except Exception:
            return

        span = abs(float(x2) - float(x1))
        length_m = max(1e-6, 2.0 * float(span))
        quarter_length_m = 0.25 * float(length_m)
        width_half_m = math.sqrt(max(0.0, float(radius) ** 2 - float(quarter_length_m) ** 2))
        width_m = 2.0 * float(width_half_m)
        if width_m <= 1e-6:
            width_m = max(1e-6, 2.0 * float(radius))

        center_offset_m = 0.5 * (float(x1) + float(x2))
        half_l = 0.5 * float(length_m)
        half_w = 0.5 * float(width_m)

        cos_psi = math.cos(float(psi))
        sin_psi = math.sin(float(psi))
        cx_m = float(x_m) + float(cos_psi) * float(center_offset_m)
        cy_m = float(y_m) + float(sin_psi) * float(center_offset_m)
        corners_m = [
            (cx_m + cos_psi * half_l - sin_psi * half_w, cy_m + sin_psi * half_l + cos_psi * half_w),
            (cx_m + cos_psi * half_l + sin_psi * half_w, cy_m + sin_psi * half_l - cos_psi * half_w),
            (cx_m - cos_psi * half_l + sin_psi * half_w, cy_m - sin_psi * half_l - cos_psi * half_w),
            (cx_m - cos_psi * half_l - sin_psi * half_w, cy_m - sin_psi * half_l + cos_psi * half_w),
        ]

        corners_px = [self.grid_to_px((float(xc) / float(cell_size_m), float(yc) / float(cell_size_m))) for xc, yc in corners_m]
        line_w = max(1, int(round(self._cell_px * 0.12)))
        color = (255, 112, 67) if bool(collision) else (79, 195, 247)
        self._pygame.draw.polygon(self._screen, color, corners_px, line_w)

    @staticmethod
    def _default_log(msg: str) -> None:
        print(str(msg), file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Interactive pygame goal-click motion planning demo (forest bicycle env).")
    ap.add_argument("--config", type=Path, default=None, help="JSON config file. CLI flags override config.")
    ap.add_argument("--profile", type=str, default=None, help="Config profile name under configs/ (game section).")

    ap.add_argument("--env", type=str, default="forest_a", choices=list(FOREST_ENV_ORDER), help="Forest map name.")
    ap.add_argument("--planner", type=_planner_from_cli, default="hybrid_astar", help="Planner: hybrid_astar rrt_star grid_astar cnn_ddqn.")
    ap.add_argument("--controller", type=str, default="mpc", choices=["mpc"], help="Controller (tracking): mpc.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600, help="Episode horizon (env max_steps).")

    ap.add_argument("--window-size", type=int, default=900)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--trail-len", type=int, default=300)
    ap.add_argument("--no-collision-box", action="store_true", help="Hide oriented collision box overlay.")

    ap.add_argument("--planner-timeout-s", type=float, default=1.0)
    ap.add_argument("--hybrid-max-nodes", type=int, default=200_000)
    ap.add_argument("--rrt-max-iter", type=int, default=5_000)
    ap.add_argument("--astar-max-expanded", type=int, default=1_000_000)
    ap.add_argument("--astar-curve-opt", action="store_true", help="Enable A* curve optimization (slower, smoother).")
    ap.add_argument("--collision-padding-m", type=float, default=0.0, help="Planner collision padding (meters).")

    ap.add_argument("--mpc-horizon-steps", type=int, default=12)
    ap.add_argument("--mpc-candidates", type=int, default=256)
    ap.add_argument("--mpc-w-pos", type=float, default=6.0)
    ap.add_argument("--mpc-w-yaw", type=float, default=1.2)

    ap.add_argument("--rl-checkpoint", type=Path, default=None, help="(cnn_ddqn planner) Path to *.pt checkpoint.")
    ap.add_argument("--rl-device", type=str, default="cpu", help="(cnn_ddqn planner) Device: cpu/cuda/auto.")
    ap.add_argument("--rl-max-steps", type=int, default=600, help="(cnn_ddqn planner) Rollout horizon.")

    ap.add_argument("--self-check", action="store_true", help="Run a fast non-rendering smoke check.")
    return ap


def _apply_game_config(ap: argparse.ArgumentParser, args: list[str]) -> argparse.Namespace:
    pre = ap.parse_args(args)
    if pre.config is None and pre.profile is None:
        return pre

    cfg_path = resolve_config_path(config=pre.config, profile=pre.profile, default_path=Path("__no_default__"))
    if cfg_path is None:
        return pre

    cfg = load_json(Path(cfg_path))
    game_cfg = select_section(cfg, section="game")
    apply_config_defaults(ap, game_cfg, strict=True)
    return ap.parse_args(args)


def _self_check(args: argparse.Namespace) -> int:
    env, grid = _build_env(env_name=str(args.env), max_steps=int(args.max_steps), seed=int(args.seed))
    grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=float(env.cell_size_m))
    ackermann = default_ackermann_params(
        wheelbase_m=float(env.model.wheelbase_m),
        delta_max_rad=float(env.model.delta_max_rad),
        v_max_m_s=float(env.model.v_max_m_s),
    )
    footprint = forest_two_circle_footprint(wheelbase_m=float(env.model.wheelbase_m))

    start_xy = (int(round(float(env._x_m) / float(env.cell_size_m))), int(round(float(env._y_m) / float(env.cell_size_m))))
    goal_xy = (int(env.goal_xy[0]), int(env.goal_xy[1]))
    start_theta = float(getattr(env, "_psi_rad", 0.0))

    res = plan_hybrid_astar(
        grid_map=grid_map,
        footprint=footprint,
        params=ackermann,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_theta_rad=float(start_theta),
        goal_theta_rad=None,
        timeout_s=max(0.05, float(args.planner_timeout_s)),
        max_nodes=int(args.hybrid_max_nodes),
        collision_padding_m=float(max(0.0, float(args.collision_padding_m))),
        goal_xy_tol_m=0.5,
        goal_theta_tol_rad=float(math.pi),
    )
    if len(res.path_xy_cells) < 2:
        print("[self-check] planner returned trivial path (ok)", file=sys.stderr)

    mpc_cfg = MPCConfig(
        horizon_steps=int(args.mpc_horizon_steps),
        candidates=int(args.mpc_candidates),
        dt_s=float(env.model.dt),
        w_pos=float(args.mpc_w_pos),
        w_yaw=float(args.mpc_w_yaw),
    )
    cell_size_m = float(env.cell_size_m)
    path_xy_m = np.asarray([(float(x) * cell_size_m, float(y) * cell_size_m) for x, y in res.path_xy_cells], dtype=np.float64)
    path_theta = (
        np.asarray([float(t) for t in (res.path_theta_rad or [])], dtype=np.float64)
        if res.path_theta_rad is not None and len(res.path_theta_rad) == int(path_xy_m.shape[0])
        else _path_headings(path_xy_m)
    )

    prev_idx = 0
    for _ in range(5):
        step = solve_mpc_one_step(env=env, path_xy_m=path_xy_m, path_theta=path_theta, prev_path_idx=int(prev_idx), config=mpc_cfg)
        prev_idx = int(step.nearest_path_idx)
        _obs, _reward, _done, _trunc, _info = env.step_continuous(delta_dot_rad_s=float(step.delta_dot_rad_s), a_m_s2=float(step.a_m_s2))

    print("[self-check] ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = _apply_game_config(ap, sys.argv[1:] if argv is None else argv)

    if bool(args.self_check):
        return _self_check(args)

    env, grid = _build_env(env_name=str(args.env), max_steps=int(args.max_steps), seed=int(args.seed))
    viewer = _PygameViewer(
        window_size=int(args.window_size),
        fps=int(args.fps),
        trail_len=int(args.trail_len),
        show_collision_box=not bool(args.no_collision_box),
    )
    if not viewer.prepare_map(grid):
        return 2

    cell_size_m = float(env.cell_size_m)
    tracker = MPCTracker(
        cfg=MPCConfig(
            horizon_steps=int(args.mpc_horizon_steps),
            candidates=int(args.mpc_candidates),
            dt_s=float(env.model.dt),
            w_pos=float(args.mpc_w_pos),
            w_yaw=float(args.mpc_w_yaw),
        )
    )

    paused = False
    plan: GamePlan | None = None
    plan_path_xy_cells: list[tuple[float, float]] | None = None
    plan_path_xy_m: np.ndarray | None = None
    plan_theta: np.ndarray | None = None
    plan_success: bool | None = None
    plan_time_s: float | None = None
    last_info: dict[str, object] | None = None

    trail: deque[tuple[float, float]] = deque(maxlen=int(args.trail_len))
    trail.append(env._agent_xy_for_plot())

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner")
    current_job: _PlanJob | None = None
    current_future: Future[GamePlan] | None = None
    request_id = 0

    def submit_plan(*, force: bool = False) -> None:
        nonlocal current_job, current_future, request_id, plan, plan_success, plan_time_s, plan_path_xy_cells, plan_path_xy_m, plan_theta
        if current_future is not None and not bool(current_future.done()) and not bool(force):
            return

        request_id += 1
        start_xy = (int(round(float(env._x_m) / cell_size_m)), int(round(float(env._y_m) / cell_size_m)))
        goal_xy = (int(env.goal_xy[0]), int(env.goal_xy[1]))
        job = _PlanJob(
            request_id=int(request_id),
            planner=_planner_from_cli(str(args.planner)),
            start_xy=start_xy,
            goal_xy=goal_xy,
            start_theta_rad=float(getattr(env, "_psi_rad", 0.0)),
            seed=int(args.seed),
        )
        current_job = job
        plan = None
        plan_success = None
        plan_time_s = None
        plan_path_xy_cells = None
        plan_path_xy_m = None
        plan_theta = None
        tracker.reset()

        current_future = executor.submit(
            _compute_plan,
            job=job,
            env_name=str(args.env),
            grid_y0_bottom=grid,
            cell_size_m=float(cell_size_m),
            planner_timeout_s=float(args.planner_timeout_s),
            hybrid_max_nodes=int(args.hybrid_max_nodes),
            rrt_max_iter=int(args.rrt_max_iter),
            astar_max_expanded=int(args.astar_max_expanded),
            astar_curve_opt=bool(args.astar_curve_opt),
            collision_padding_m=float(args.collision_padding_m),
            rl_checkpoint=(Path(args.rl_checkpoint) if args.rl_checkpoint is not None else None),
            rl_device=str(select_device(device=str(args.rl_device))),
            rl_max_steps=int(args.rl_max_steps),
        )

    def apply_plan(res: GamePlan) -> None:
        nonlocal plan, plan_success, plan_time_s, plan_path_xy_cells, plan_path_xy_m, plan_theta
        plan = res
        plan_success = bool(res.success)
        plan_time_s = float(res.time_s)
        plan_path_xy_cells = list(res.path_xy_cells)
        plan_path_xy_m = np.asarray([(float(x) * cell_size_m, float(y) * cell_size_m) for x, y in res.path_xy_cells], dtype=np.float64)

        if res.path_theta_rad is not None and len(res.path_theta_rad) == int(plan_path_xy_m.shape[0]):
            plan_theta = np.asarray([float(t) for t in res.path_theta_rad], dtype=np.float64)
        else:
            plan_theta = _path_headings(plan_path_xy_m)
        tracker.reset()

    def reset_episode() -> None:
        nonlocal paused, last_info
        env.reset(seed=int(args.seed))
        trail.clear()
        trail.append(env._agent_xy_for_plot())
        last_info = None
        paused = False
        submit_plan(force=True)
        viewer.set_status("已重置：重新开始")

    def set_goal_from_click(cell_xy: tuple[int, int]) -> None:
        nonlocal paused, last_info
        snapped = _snap_to_free_cell(grid, xy=cell_xy, max_radius=12)
        if snapped is None:
            viewer.set_status("目标点：越界")
            return

        gx, gy = int(snapped[0]), int(snapped[1])
        if bool(grid[gy, gx]):
            viewer.set_status("目标点：被障碍占用")
            return

        try:
            env._set_goal_xy((int(gx), int(gy)))
            sx = int(round(float(env._x_m) / cell_size_m))
            sy = int(round(float(env._y_m) / cell_size_m))
            env._update_start_dependent_fields(start_xy=(int(sx), int(sy)))
        except Exception as exc:
            viewer.set_status(f"设置目标失败：{exc}")
            return

        paused = False
        last_info = None
        submit_plan(force=True)
        viewer.set_status(f"目标点：({gx},{gy})，开始规划…")

    submit_plan(force=True)

    try:
        while viewer.running:
            events = viewer.poll_events()
            if not viewer.running:
                break

            if viewer._pygame is None:
                break
            pg = viewer._pygame

            for ev in events:
                if ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_SPACE:
                        paused = not bool(paused)
                    elif ev.key == pg.K_r:
                        reset_episode()
                    elif ev.key == pg.K_p:
                        submit_plan(force=True)
                    elif ev.key == pg.K_1:
                        args.planner = "hybrid_astar"
                        submit_plan(force=True)
                    elif ev.key == pg.K_2:
                        args.planner = "rrt_star"
                        submit_plan(force=True)
                    elif ev.key == pg.K_3:
                        args.planner = "grid_astar"
                        submit_plan(force=True)
                    elif ev.key == pg.K_4:
                        args.planner = "cnn_ddqn"
                        submit_plan(force=True)
                elif ev.type == pg.MOUSEBUTTONDOWN and getattr(ev, "button", None) == 1:
                    pos = getattr(ev, "pos", None)
                    if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                        grid_xy = viewer.px_to_grid(int(pos[0]), int(pos[1]))
                        if grid_xy is not None:
                            cell_xy = (int(round(float(grid_xy[0]))), int(round(float(grid_xy[1]))))
                            set_goal_from_click(cell_xy)

            if current_future is not None and bool(current_future.done()) and current_job is not None:
                try:
                    res = current_future.result()
                except Exception as exc:
                    viewer.set_status(f"规划器错误：{exc}")
                    current_future = None
                    current_job = None
                else:
                    apply_plan(res)
                    viewer.set_status(f"规划完成：{res.planner}  成功={int(res.success)}  用时={res.time_s:.3f}s")
                    current_future = None
                    current_job = None

            if not bool(paused) and plan_path_xy_m is not None and plan_theta is not None and plan_path_xy_m.shape[0] >= 2:
                step = solve_mpc_one_step(
                    env=env,
                    path_xy_m=plan_path_xy_m,
                    path_theta=plan_theta,
                    prev_path_idx=int(tracker.prev_path_idx),
                    config=tracker.cfg,
                )
                tracker.prev_path_idx = int(step.nearest_path_idx)
                a_cmd = float(step.a_m_s2) if bool(step.feasible) else 0.0
                dd_cmd = float(step.delta_dot_rad_s) if bool(step.feasible) else 0.0
                _obs, _reward, done, trunc, info = env.step_continuous(delta_dot_rad_s=float(dd_cmd), a_m_s2=float(a_cmd))
                last_info = dict(info)
                trail.append(env._agent_xy_for_plot())

                if bool(info.get("reached", False)):
                    paused = True
                    viewer.set_status("到达目标：已暂停（再点一个新目标继续）")
                elif bool(info.get("collision", False) or info.get("stuck", False)):
                    paused = True
                    viewer.set_status("发生碰撞/卡住：已暂停（按 R 重置）")
                elif bool(done) or bool(trunc):
                    paused = True
                    viewer.set_status("结束/超时：已暂停（按 R 重置）")

            viewer.render(
                env_name=str(args.env),
                planner=str(args.planner),
                controller=str(args.controller),
                paused=bool(paused),
                env=env,
                start_xy=(int(env.start_xy[0]), int(env.start_xy[1])),
                goal_xy=(int(env.goal_xy[0]), int(env.goal_xy[1])),
                trail=trail,
                plan_path_xy_cells=plan_path_xy_cells,
                plan_success=plan_success,
                plan_time_s=plan_time_s,
                info=last_info,
            )
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
