from __future__ import annotations

import math
import sys
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


class TrainLiveViewer:
    def __init__(
        self,
        *,
        enabled: bool = False,
        fps: int = 0,
        window_size: int = 900,
        trail_len: int = 300,
        skip_steps: int = 1,
        show_collision_box: bool = True,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.fps = max(0, int(fps))
        self.window_size = max(64, int(window_size))
        self.trail_len = max(1, int(trail_len))
        self.skip_steps = max(1, int(skip_steps))
        self.show_collision_box = bool(show_collision_box)
        self._log = logger if logger is not None else self._default_log

        self._disabled_reason: str | None = None
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

        self._env_name: str = ""
        self._algo: str = ""
        self._episode: int = 0
        self._total_episodes: int = 0
        self._step: int = 0
        self._action: int = 0
        self._reward: float = 0.0
        self._done: bool = False
        self._truncated: bool = False
        self._collision: bool = False
        self._reached: bool = False

        self._start_xy: tuple[float, float] = (0.0, 0.0)
        self._goal_xy: tuple[float, float] = (0.0, 0.0)
        self._trail: deque[tuple[float, float]] = deque(maxlen=self.trail_len)
        self._env_ref: Any | None = None
        self._pose_m: tuple[float, float, float] | None = None
        self._box_center_offset_m: float = 0.0
        self._box_half_length_m: float = 0.0
        self._box_half_width_m: float = 0.0
        self._started = False

    @property
    def disabled(self) -> bool:
        return (not self.enabled) or (self._disabled_reason is not None)

    def start_episode(
        self,
        *,
        env: Any,
        env_name: str,
        algo: str,
        episode: int,
        total_episodes: int,
        info: Mapping[str, object] | None = None,
    ) -> None:
        if not self.enabled or self.disabled:
            return
        if not self._ensure_backend():
            return
        if not self._prepare_map(env):
            return

        self._env_name = str(env_name)
        self._algo = str(algo)
        self._episode = int(episode)
        self._total_episodes = max(1, int(total_episodes))
        self._step = 0
        self._action = 0
        self._reward = 0.0
        self._done = False
        self._truncated = False
        self._collision = False
        self._reached = False

        self._start_xy = (float(getattr(env, "start_xy", (0, 0))[0]), float(getattr(env, "start_xy", (0, 0))[1]))
        self._goal_xy = (float(getattr(env, "goal_xy", (0, 0))[0]), float(getattr(env, "goal_xy", (0, 0))[1]))
        self._env_ref = env
        self._setup_collision_box_geometry(env)

        self._trail.clear()
        start_xy = self._extract_xy(info, default_xy=self._start_xy)
        self._trail.append(start_xy)
        self._update_pose_from_info(info=info, env=env, fallback_xy=start_xy)
        self._started = True
        self._draw_frame()

    def update_step(
        self,
        *,
        env: Any,
        env_name: str,
        algo: str,
        episode: int,
        total_episodes: int,
        step: int,
        action: int,
        reward: float,
        done: bool,
        truncated: bool,
        info: Mapping[str, object] | None,
    ) -> None:
        if not self.enabled or self.disabled:
            return
        if not self._ensure_backend():
            return
        if not self._started:
            self.start_episode(
                env=env,
                env_name=env_name,
                algo=algo,
                episode=episode,
                total_episodes=total_episodes,
                info=info,
            )
            if self.disabled:
                return

        self._handle_events()
        if self.disabled:
            return

        self._step = int(step)
        self._action = int(action)
        self._reward = float(reward)
        self._done = bool(done)
        self._truncated = bool(truncated)
        self._collision = bool((info or {}).get("collision", False))
        self._reached = bool((info or {}).get("reached", False))
        self._env_name = str(env_name)
        self._algo = str(algo)
        self._episode = int(episode)
        self._total_episodes = max(1, int(total_episodes))
        self._env_ref = env

        xy = self._extract_xy(info, default_xy=self._trail[-1] if self._trail else self._start_xy)
        self._trail.append(xy)
        self._update_pose_from_info(info=info, env=env, fallback_xy=xy)

        should_draw = (self._step % self.skip_steps) == 0 or self._done or self._truncated
        if should_draw:
            self._draw_frame()

        if self._clock is not None and self.fps > 0:
            self._clock.tick(self.fps)

    def close(self) -> None:
        self._started = False
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
            self._env_ref = None
            self._pose_m = None

    def _prepare_map(self, env: Any) -> bool:
        grid = getattr(env, "grid", None)
        if grid is None:
            self._disable("live-view disabled: env has no grid attribute")
            return False
        grid_np = np.asarray(grid)
        if grid_np.ndim != 2:
            self._disable("live-view disabled: grid must be 2D")
            return False

        shape = (int(grid_np.shape[0]), int(grid_np.shape[1]))
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
        obs = np.argwhere(grid_np.astype(bool, copy=False))
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

    def _draw_frame(self) -> None:
        if self._screen is None or self._pygame is None:
            return

        self._handle_events()
        if self.disabled:
            return

        self._screen.fill((18, 20, 24))
        if self._map_surface is not None:
            self._screen.blit(self._map_surface, (self._map_left, self._map_top))

        if len(self._trail) >= 2:
            trail_points = [self._grid_to_px(xy) for xy in self._trail]
            self._pygame.draw.lines(self._screen, (255, 191, 0), False, trail_points, 2)

        start_px = self._grid_to_px(self._start_xy)
        goal_px = self._grid_to_px(self._goal_xy)
        cur_xy = self._trail[-1] if self._trail else self._start_xy
        agent_px = self._grid_to_px(cur_xy)

        radius = max(3, int(round(self._cell_px * 0.28)))
        self._pygame.draw.circle(self._screen, (66, 165, 245), start_px, radius + 1)
        self._pygame.draw.circle(self._screen, (239, 83, 80), goal_px, radius + 1)

        if self.show_collision_box and self._env_ref is not None:
            self._draw_oriented_collision_box(env=self._env_ref)

        agent_color = (102, 187, 106)
        if self._collision:
            agent_color = (255, 112, 67)
        elif self._reached:
            agent_color = (171, 71, 188)
        self._pygame.draw.circle(self._screen, agent_color, agent_px, radius + 2)

        self._draw_text()
        try:
            self._pygame.display.set_caption(
                f"train live: {self._env_name}/{self._algo} ep {self._episode}/{self._total_episodes} step {self._step}"
            )
        except Exception:
            pass
        self._pygame.display.flip()

    def _draw_text(self) -> None:
        if self._screen is None or self._font is None:
            return
        lines = [
            f"{self._env_name} | {self._algo}",
            f"ep {self._episode}/{self._total_episodes}  step {self._step}  action {self._action}",
            f"reward {self._reward:+.3f}  reached={int(self._reached)}  collision={int(self._collision)}  done={int(self._done)}",
        ]
        y = 8
        for line in lines:
            surf = self._font.render(line, True, (240, 240, 240))
            self._screen.blit(surf, (10, y))
            y += max(18, int(surf.get_height()) + 2)

    def _handle_events(self) -> None:
        if self._pygame is None:
            return
        try:
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self._disable("live-view window closed by user; training continues")
                    break
        except Exception as exc:
            self._disable(f"live-view event handling failed: {exc}")

    def _extract_xy(self, info: Mapping[str, object] | None, *, default_xy: tuple[float, float]) -> tuple[float, float]:
        if info is None:
            return default_xy
        raw = info.get("agent_xy")
        if isinstance(raw, (tuple, list)) and len(raw) >= 2:
            try:
                return (float(raw[0]), float(raw[1]))
            except Exception:
                return default_xy
        return default_xy

    def _update_pose_from_info(
        self,
        *,
        info: Mapping[str, object] | None,
        env: Any,
        fallback_xy: tuple[float, float],
    ) -> None:
        if info is not None:
            pose = info.get("pose_m")
            if isinstance(pose, (tuple, list)) and len(pose) >= 3:
                try:
                    self._pose_m = (float(pose[0]), float(pose[1]), float(pose[2]))
                    return
                except Exception:
                    pass

        if all(hasattr(env, attr) for attr in ("_x_m", "_y_m", "_psi_rad")):
            try:
                self._pose_m = (float(env._x_m), float(env._y_m), float(env._psi_rad))
                return
            except Exception:
                pass

        cell_size_m = self._cell_size_m(env)
        psi = 0.0
        if hasattr(env, "_psi_rad"):
            try:
                psi = float(env._psi_rad)
            except Exception:
                psi = 0.0
        self._pose_m = (float(fallback_xy[0]) * float(cell_size_m), float(fallback_xy[1]) * float(cell_size_m), psi)

    def _setup_collision_box_geometry(self, env: Any) -> None:
        cell_size_m = max(1e-6, float(self._cell_size_m(env)))
        center_offset_m = 0.0
        half_length_m = 0.5 * cell_size_m
        half_width_m = 0.35 * cell_size_m

        footprint = getattr(env, "footprint", None)
        if footprint is not None:
            try:
                x1 = float(getattr(footprint, "x1_m"))
                x2 = float(getattr(footprint, "x2_m"))
                radius = float(getattr(footprint, "radius_m"))
                span = abs(float(x2) - float(x1))
                length_m = max(1e-6, 2.0 * float(span))
                quarter_length_m = 0.25 * float(length_m)
                width_half_m = math.sqrt(max(0.0, float(radius) ** 2 - float(quarter_length_m) ** 2))
                width_m = 2.0 * float(width_half_m)
                if width_m <= 1e-6:
                    width_m = max(1e-6, 2.0 * float(radius))

                center_offset_m = 0.5 * (float(x1) + float(x2))
                half_length_m = 0.5 * float(length_m)
                half_width_m = 0.5 * float(width_m)

                pad_m = max(0.0, float(getattr(env, "_eps_cell_m", 0.0)))
                half_length_m += float(pad_m)
                half_width_m += float(pad_m)
            except Exception:
                pass

        self._box_center_offset_m = float(center_offset_m)
        self._box_half_length_m = float(max(1e-6, half_length_m))
        self._box_half_width_m = float(max(1e-6, half_width_m))

    def _draw_oriented_collision_box(self, *, env: Any) -> None:
        if self._pygame is None or self._screen is None or self._pose_m is None:
            return

        x_m, y_m, psi = self._pose_m
        cos_psi = math.cos(float(psi))
        sin_psi = math.sin(float(psi))
        cx_m = float(x_m) + float(cos_psi) * float(self._box_center_offset_m)
        cy_m = float(y_m) + float(sin_psi) * float(self._box_center_offset_m)

        half_l = float(self._box_half_length_m)
        half_w = float(self._box_half_width_m)
        corners_m = [
            (cx_m + cos_psi * half_l - sin_psi * half_w, cy_m + sin_psi * half_l + cos_psi * half_w),
            (cx_m + cos_psi * half_l + sin_psi * half_w, cy_m + sin_psi * half_l - cos_psi * half_w),
            (cx_m - cos_psi * half_l + sin_psi * half_w, cy_m - sin_psi * half_l - cos_psi * half_w),
            (cx_m - cos_psi * half_l - sin_psi * half_w, cy_m - sin_psi * half_l + cos_psi * half_w),
        ]

        cell_size_m = max(1e-6, float(self._cell_size_m(env)))
        corners_px = [self._grid_to_px((float(xc) / float(cell_size_m), float(yc) / float(cell_size_m))) for xc, yc in corners_m]
        line_w = max(1, int(round(self._cell_px * 0.12)))
        color = (255, 112, 67) if self._collision else (79, 195, 247)
        self._pygame.draw.polygon(self._screen, color, corners_px, line_w)

    @staticmethod
    def _cell_size_m(env: Any) -> float:
        if hasattr(env, "cell_size_m"):
            try:
                return float(env.cell_size_m)
            except Exception:
                pass
        if hasattr(env, "cell_size"):
            try:
                return float(env.cell_size)
            except Exception:
                pass
        return 1.0

    def _grid_to_px(self, xy: tuple[float, float]) -> tuple[int, int]:
        h = int(self._map_shape[0]) if self._map_shape is not None else 1
        x = float(xy[0])
        y = float(xy[1])
        px = float(self._map_left) + (x + 0.5) * float(self._cell_px)
        py = float(self._map_top) + (float(h) - (y + 0.5)) * float(self._cell_px)
        return (int(round(px)), int(round(py)))

    def _ensure_backend(self) -> bool:
        if self.disabled:
            return False

        if self._pygame is None:
            try:
                import pygame  # type: ignore
            except Exception:
                self._disable(
                    "live-view enabled but pygame is missing; install via `pip install -r requirements-optional.txt`"
                )
                return False
            self._pygame = pygame

        if self._screen is not None:
            return True

        try:
            self._pygame.init()
            self._pygame.font.init()
            self._screen = self._pygame.display.set_mode((self.window_size, self.window_size))
            self._clock = self._pygame.time.Clock()
            self._font = self._pygame.font.SysFont("Consolas", 16)
            if self._font is None:
                self._font = self._pygame.font.Font(None, 18)
            return True
        except Exception as exc:
            self._disable(f"live-view init failed: {exc}")
            return False

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            self._disabled_reason = str(reason)
            self._log(f"[train] {self._disabled_reason}")
        self.close()

    @staticmethod
    def _default_log(msg: str) -> None:
        print(str(msg), file=sys.stderr, flush=True)
