from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from amr_dqn.runtime import configure_runtime, select_device, torch_runtime_info
from amr_dqn.runs import create_run_dir, resolve_experiment_dir, resolve_models_dir

configure_runtime()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import gym
import numpy as np
import pandas as pd
import torch

from amr_dqn.agents import AgentConfig, DQNFamilyAgent
from amr_dqn.baselines.pathplan import (
    default_ackermann_params,
    forest_oriented_box_footprint,
    grid_map_from_obstacles,
    plan_hybrid_astar,
    plan_rrt_star,
    point_footprint,
)
from amr_dqn.env import AMRBicycleEnv, AMRGridEnv, RewardWeights, bicycle_integrate_one_step, wrap_angle_rad
from amr_dqn.maps import ENV_ORDER, FOREST_ENV_ORDER, get_map_spec
from amr_dqn.metrics import KPI, max_corner_degree, min_distance_to_obstacle, num_path_corners, path_length
from amr_dqn.smoothing import chaikin_smooth


@dataclass(frozen=True)
class PathTrace:
    path_xy_cells: list[tuple[float, float]]
    success: bool


def rollout_agent(
    env: gym.Env,
    agent: DQNFamilyAgent,
    *,
    max_steps: int,
    seed: int,
    time_mode: str = "rollout",
    obs_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    forest_adm_horizon: int = 15,
    forest_topk: int = 10,
    forest_min_od_m: float = 0.0,
    forest_min_progress_m: float = 1e-4,
) -> tuple[list[tuple[float, float]], float, bool]:
    obs, _ = env.reset(seed=seed)
    if obs_transform is not None:
        obs = obs_transform(obs)
    path: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]

    def sync_cuda() -> None:
        if agent.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    time_mode = str(time_mode).lower().strip()
    if time_mode not in {"rollout", "policy"}:
        raise ValueError("time_mode must be one of: rollout, policy")

    inference_time_s = 0.0
    sync_cuda()
    t_rollout0 = time.perf_counter()
    done = False
    truncated = False
    steps = 0
    reached = False
    adm_h = max(1, int(forest_adm_horizon))
    topk_k = max(1, int(forest_topk))
    min_od = float(forest_min_od_m)
    min_prog = float(forest_min_progress_m)
    while not (done or truncated) and steps < max_steps:
        steps += 1
        if time_mode == "policy":
            sync_cuda()
            t0 = time.perf_counter()
        if isinstance(env, AMRBicycleEnv):
            # Forest policy rollout: evaluate the greedy policy, but only compute the expensive
            # admissible-action mask when needed. For performance, reuse a single Q forward-pass
            # for greedy selection, top-k fallback, and masked selection.
            with torch.no_grad():
                x = torch.from_numpy(obs.astype(np.float32, copy=False)).to(agent.device)
                q = agent.q(x.unsqueeze(0)).squeeze(0)
                a = int(torch.argmax(q).item())

            if not bool(env.is_action_admissible(int(a), horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog)):
                chosen: int | None = None
                kk = int(min(int(topk_k), int(q.numel())))
                topk = torch.topk(q, k=kk, dim=0).indices.detach().cpu().numpy()
                for cand in topk.tolist():
                    cand_i = int(cand)
                    if cand_i == int(a):
                        continue
                    if bool(env.is_action_admissible(cand_i, horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog)):
                        chosen = cand_i
                        break

                if chosen is None:
                    mask = env.admissible_action_mask(horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog)
                    if bool(mask.any()):
                        q_masked = q.clone()
                        q_masked[torch.from_numpy(~mask).to(q.device)] = torch.finfo(q_masked.dtype).min
                        chosen = int(torch.argmax(q_masked).item())

                if chosen is not None:
                    a = int(chosen)
        else:
            a = agent.act(obs, episode=0, explore=False)
        if time_mode == "policy":
            sync_cuda()
            inference_time_s += float(time.perf_counter() - t0)
        obs, _, done, truncated, info = env.step(a)
        if obs_transform is not None:
            obs = obs_transform(obs)
        x, y = info["agent_xy"]
        path.append((float(x), float(y)))
        if info.get("reached"):
            reached = True
            break
    return path, float(inference_time_s), bool(reached)


def rollout_tracked_path(
    env: AMRBicycleEnv,
    ref_path_xy_cells: list[tuple[float, float]],
    *,
    max_steps: int,
    seed: int,
    lookahead_points: int = 5,
    horizon_steps: int = 15,
    w_target: float = 0.2,
    w_heading: float = 0.2,
    w_clearance: float = 0.8,
    w_speed: float = 0.0,
) -> tuple[list[tuple[float, float]], float, bool]:
    obs, _ = env.reset(seed=seed)
    path: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]

    def choose_action(progress_idx: int) -> tuple[int, int]:
        if len(ref_path_xy_cells) < 2:
            return int(env.expert_action_mpc(horizon_steps=int(horizon_steps), w_clearance=float(w_clearance), w_speed=float(w_speed))), int(
                progress_idx
            )

        x_cells = float(env._x_m) / float(env.cell_size_m)
        y_cells = float(env._y_m) / float(env.cell_size_m)

        # Find nearest reference-path index (windowed search around previous index).
        start_i = max(0, int(progress_idx) - 25)
        end_i = min(len(ref_path_xy_cells), int(progress_idx) + 250)
        if end_i <= start_i:
            start_i, end_i = 0, len(ref_path_xy_cells)
        best_i = start_i
        best_d2 = float("inf")
        for i in range(start_i, end_i):
            px, py = ref_path_xy_cells[i]
            d2 = (float(px) - x_cells) ** 2 + (float(py) - y_cells) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        progress_idx = int(best_i)

        la = max(1, int(lookahead_points))
        tgt_i = min(int(best_i) + la, len(ref_path_xy_cells) - 1)
        tx_cells, ty_cells = ref_path_xy_cells[tgt_i]
        tx_m = float(tx_cells) * float(env.cell_size_m)
        ty_m = float(ty_cells) * float(env.cell_size_m)

        h = max(1, int(horizon_steps))
        best_score = -float("inf")
        best_action: int | None = None

        x0 = float(env._x_m)
        y0 = float(env._y_m)
        psi0 = float(env._psi_rad)
        v0 = float(env._v_m_s)
        delta0 = float(env._delta_rad)
        v_max = float(env.model.v_max_m_s)

        for a_id in range(int(env.action_table.shape[0])):
            delta_dot = float(env.action_table[a_id, 0])
            a = float(env.action_table[a_id, 1])

            x, y, psi, v, delta = x0, y0, psi0, v0, delta0
            min_od = float("inf")
            ok = True
            for _ in range(h):
                x, y, psi, v, delta = bicycle_integrate_one_step(
                    x_m=x,
                    y_m=y,
                    psi_rad=psi,
                    v_m_s=v,
                    delta_rad=delta,
                    delta_dot_rad_s=delta_dot,
                    a_m_s2=a,
                    params=env.model,
                )
                od, coll = env._od_and_collision_at_pose_m(x, y, psi)
                min_od = min(float(min_od), float(od))
                if coll:
                    ok = False
                    break
            if not ok:
                continue

            cost = float(env._cost_to_goal_pose_m(x, y, psi))
            dist_tgt = float(math.hypot(float(tx_m) - float(x), float(ty_m) - float(y)))
            tgt_heading = float(math.atan2(float(ty_m) - float(y), float(tx_m) - float(x)))
            heading_err = wrap_angle_rad(float(tgt_heading) - float(psi))

            score = -float(cost)
            score += -float(w_target) * float(dist_tgt) - float(w_heading) * abs(float(heading_err))
            score += float(w_clearance) * float(min_od)
            if w_speed:
                score += float(w_speed) * (float(v) / float(v_max))

            if float(score) > float(best_score):
                best_score = float(score)
                best_action = int(a_id)

        if best_action is None or not math.isfinite(best_score):
            return int(env.expert_action_mpc(horizon_steps=int(horizon_steps), w_clearance=float(w_clearance), w_speed=float(w_speed))), int(
                progress_idx
            )

        return int(best_action), int(progress_idx)

    inference_time_s = 0.0
    done = False
    truncated = False
    steps = 0
    reached = False
    progress_idx = 0
    while not (done or truncated) and steps < max_steps:
        steps += 1
        t0 = time.perf_counter()
        a, progress_idx = choose_action(progress_idx)
        inference_time_s += float(time.perf_counter() - t0)
        obs, _, done, truncated, info = env.step(int(a))
        x, y = info["agent_xy"]
        path.append((float(x), float(y)))
        if info.get("reached"):
            reached = True
            break
    if time_mode == "rollout":
        sync_cuda()
        inference_time_s = float(time.perf_counter() - t_rollout0)

    return path, float(inference_time_s), bool(reached)


def infer_checkpoint_obs_dim(path: Path) -> int:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or "q_state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    sd = payload["q_state_dict"]
    w = sd.get("net.0.weight")
    if w is None:
        w = sd.get("feature.0.weight")
    if w is None:
        raise ValueError(f"Could not infer observation dim from checkpoint: {path}")
    return int(w.shape[1])


def forest_legacy_obs_transform(obs: np.ndarray) -> np.ndarray:
    """Map current forest observations (11+n_sectors) -> legacy (7+n_sectors)."""
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    if x.size < 11:
        return x
    return np.concatenate([x[:7], x[11:]]).astype(np.float32, copy=False)


def mean_kpi(kpis: list[KPI]) -> KPI:
    if not kpis:
        nan = float("nan")
        return KPI(
            avg_path_length=nan,
            num_corners=nan,
            min_collision_dist=nan,
            inference_time_s=nan,
            max_corner_deg=nan,
        )
    return KPI(
        avg_path_length=float(np.mean([k.avg_path_length for k in kpis])),
        num_corners=float(np.mean([k.num_corners for k in kpis])),
        min_collision_dist=float(np.mean([k.min_collision_dist for k in kpis])),
        inference_time_s=float(np.mean([k.inference_time_s for k in kpis])),
        max_corner_deg=float(np.mean([k.max_corner_deg for k in kpis])),
    )


def smooth_path(path: list[tuple[float, float]], *, iterations: int) -> list[tuple[float, float]]:
    if not path:
        return []
    pts = np.array(path, dtype=np.float32)
    sm = chaikin_smooth(pts, iterations=max(0, int(iterations)))
    return [(float(x), float(y)) for x, y in sm]


def plot_env(ax: plt.Axes, grid: np.ndarray, *, title: str) -> None:
    ax.imshow(grid, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect("equal")
    h, w = grid.shape
    size = int(max(h, w))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="x", labelrotation=45)

    if size <= 60:
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(True, which="minor", alpha=0.18, linewidth=0.35)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    else:
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(False, which="minor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def draw_vehicle_boxes(
    ax: plt.Axes,
    trace: PathTrace,
    *,
    length_cells: float,
    width_cells: float,
    color: str,
) -> None:
    if not (float(length_cells) > 0.0 and float(width_cells) > 0.0):
        return
    path = trace.path_xy_cells
    if len(path) < 2:
        return

    stride = max(1, int(len(path) / 18))
    hl = 0.5 * float(length_cells)
    hw = 0.5 * float(width_cells)
    alpha = 0.28 if trace.success else 0.18
    ls = "-" if trace.success else ":"

    prev_theta: float | None = None
    for i in range(0, len(path), stride):
        x, y = path[i]
        if i < len(path) - 1:
            x2, y2 = path[i + 1]
            dx = float(x2) - float(x)
            dy = float(y2) - float(y)
        else:
            x2, y2 = path[i - 1]
            dx = float(x) - float(x2)
            dy = float(y) - float(y2)

        if abs(dx) + abs(dy) < 1e-9:
            theta = float(prev_theta) if prev_theta is not None else 0.0
        else:
            theta = float(math.atan2(dy, dx))
        prev_theta = float(theta)

        c = float(math.cos(theta))
        s = float(math.sin(theta))
        corners = [
            (float(x) + c * hl - s * hw, float(y) + s * hl + c * hw),
            (float(x) + c * hl - s * (-hw), float(y) + s * hl + c * (-hw)),
            (float(x) + c * (-hl) - s * (-hw), float(y) + s * (-hl) + c * (-hw)),
            (float(x) + c * (-hl) - s * hw, float(y) + s * (-hl) + c * hw),
        ]
        poly = mpatches.Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=0.6,
            alpha=float(alpha),
            linestyle=ls,
        )
        ax.add_patch(poly)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference and generate Fig.12 + Table II-style KPIs.")
    ap.add_argument(
        "--envs",
        nargs="*",
        default=list(ENV_ORDER),
        help="Subset of envs: a b c d forest_a forest_b forest_c forest_d",
    )
    ap.add_argument(
        "--models",
        type=Path,
        default=Path("outputs"),
        help="Model source: experiment name/dir, run dir, or models dir.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help=(
            "Output experiment name/dir. If this resolves to the same experiment as --models, "
            "results are stored under that training run directory."
        ),
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="If --out/--models is a bare name, store/read it under this directory.",
    )
    ap.add_argument(
        "--timestamp-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write into <experiment>/<timestamp>/ (or <train_run>/infer/<timestamp>/) to avoid mixing outputs.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5, help="Averaging runs for stochastic methods.")
    ap.add_argument(
        "--baselines",
        nargs="*",
        default=[],
        help="Optional baselines to evaluate: hybrid_astar rrt_star (or 'all'). Default: none.",
    )
    ap.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip loading/running DQN+IDDQN (useful for baseline-only evaluation).",
    )
    ap.add_argument("--baseline-timeout", type=float, default=5.0, help="Planner timeout (seconds).")
    ap.add_argument("--hybrid-max-nodes", type=int, default=200_000, help="Hybrid A* node budget.")
    ap.add_argument("--rrt-max-iter", type=int, default=5_000, help="RRT* iteration budget.")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--sensor-range", type=int, default=6)
    ap.add_argument(
        "--n-sectors",
        type=int,
        default=36,
        help="Forest lidar sectors (kept for backwards compatibility; not used by the global-map observation).",
    )
    ap.add_argument(
        "--obs-map-size",
        type=int,
        default=12,
        help="Downsampled global-map observation size (applies to both grid and forest envs).",
    )
    ap.add_argument("--cell-size", type=float, default=1.0, help="Grid cell size in meters.")
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device selection (default: auto).",
    )
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (when using --device=cuda).")
    ap.add_argument(
        "--score-time-weight",
        type=float,
        default=0.5,
        help=(
            "Time weight (m/s) for the composite planning_cost metric: "
            "planning_cost = (avg_path_length + w * inference_time_s) / max(success_rate, eps)."
        ),
    )
    ap.add_argument(
        "--kpi-time-mode",
        choices=("rollout", "policy"),
        default="rollout",
        help=(
            "How to measure inference_time_s for RL rollouts. "
            "'rollout' includes the full rollout wall-clock time (including env.step); "
            "'policy' measures only action-selection compute time (Q forward + admissibility checks)."
        ),
    )
    ap.add_argument(
        "--forest-adm-horizon",
        type=int,
        default=15,
        help="Forest-only: admissible-action horizon steps for safe/progress-gated rollouts.",
    )
    ap.add_argument(
        "--forest-topk",
        type=int,
        default=10,
        help="Forest-only: try the top-k greedy actions before computing a full admissible-action mask.",
    )
    ap.add_argument(
        "--forest-min-progress-m",
        type=float,
        default=1e-4,
        help="Forest-only: minimum cost-to-go progress required by admissible-action gating.",
    )
    ap.add_argument(
        "--forest-min-od-m",
        type=float,
        default=0.0,
        help="Forest-only: minimum clearance (OD) required by admissible-action gating.",
    )
    ap.add_argument(
        "--forest-baseline-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when baselines are enabled, roll out a discrete-action tracker on the planned "
            "baseline path to report executed-trajectory KPIs. Adds the tracker compute time to inference_time_s."
        ),
    )
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Print CUDA/runtime info and exit (use to verify CUDA setup).",
    )
    args = ap.parse_args()

    baseline_aliases = {
        "hybrid_astar": "hybrid_astar",
        "hybrid-a-star": "hybrid_astar",
        "hybrid": "hybrid_astar",
        "ha": "hybrid_astar",
        "rrt_star": "rrt_star",
        "rrt*": "rrt_star",
        "rrt": "rrt_star",
        "ss-rrt*": "rrt_star",
        "ss_rrt_star": "rrt_star",
        "ss_rrt*": "rrt_star",
        "all": "all",
    }
    baselines: list[str] = []
    for raw in args.baselines:
        key = str(raw).strip().lower()
        if not key:
            continue
        mapped = baseline_aliases.get(key)
        if mapped is None:
            raise SystemExit(
                f"Unknown baseline {raw!r}. Options: hybrid_astar, rrt_star, all (aliases: hybrid, rrt, rrt*)."
            )
        if mapped == "all":
            baselines = ["hybrid_astar", "rrt_star"]
            break
        if mapped not in baselines:
            baselines.append(mapped)

    if bool(args.skip_rl) and not baselines:
        raise SystemExit("--skip-rl requires at least one baseline via --baselines (e.g., --baselines all).")

    if args.self_check:
        info = torch_runtime_info()
        print(f"torch={info.torch_version}")
        print(f"cuda_available={info.cuda_available}")
        print(f"torch_cuda_version={info.torch_cuda_version}")
        print(f"cuda_device_count={info.device_count}")
        if info.device_names:
            print("cuda_devices=" + ", ".join(info.device_names))
        try:
            device = select_device(device=args.device, cuda_device=args.cuda_device)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"device_ok={device}")
        return 0

    try:
        device = select_device(device=args.device, cuda_device=args.cuda_device)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    requested_experiment_dir = resolve_experiment_dir(args.out, runs_root=args.runs_root)
    models_dir: Path | None = None
    if not bool(args.skip_rl):
        models_dir = resolve_models_dir(args.models, runs_root=args.runs_root)

        models_run_dir = models_dir.parent
        models_experiment_dir = models_run_dir.parent

        # If the output points to the same experiment (timestamped runs) or the same run dir (no-timestamp runs),
        # keep inference outputs attached to the training run.
        requested_resolved = requested_experiment_dir.resolve(strict=False)
        models_run_resolved = models_run_dir.resolve(strict=False)
        models_experiment_resolved = models_experiment_dir.resolve(strict=False)

        if requested_resolved == models_run_resolved or requested_resolved == models_experiment_resolved:
            # Keep inference outputs attached to the training run to avoid creating a sibling timestamped run.
            experiment_dir = models_run_dir / "infer"
        else:
            experiment_dir = requested_experiment_dir
    else:
        experiment_dir = requested_experiment_dir

    run_paths = create_run_dir(experiment_dir, timestamp_runs=args.timestamp_runs)
    out_dir = run_paths.run_dir

    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    args_payload: dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_payload[k] = str(v)
        else:
            args_payload[k] = v
    (out_dir / "configs" / "run.json").write_text(
        json.dumps(
            {
                "kind": "infer",
                "argv": list(sys.argv),
                "experiment_dir": str(run_paths.experiment_dir),
                "run_dir": str(run_paths.run_dir),
                "models_dir": (str(models_dir) if models_dir is not None else None),
                "args": args_payload,
                "torch": asdict(torch_runtime_info()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    paths_for_plot: dict[str, dict[str, PathTrace]] = {}
    plot_meta: dict[str, dict[str, float]] = {}

    for env_name in args.envs:
        spec = get_map_spec(env_name)
        if env_name in FOREST_ENV_ORDER:
            env = AMRBicycleEnv(
                spec,
                max_steps=args.max_steps,
                cell_size_m=0.1,
                sensor_range_m=float(args.sensor_range),
                n_sectors=args.n_sectors,
                obs_map_size=int(args.obs_map_size),
            )
            cell_size_m = 0.1
        else:
            env = AMRGridEnv(
                spec,
                sensor_range=args.sensor_range,
                max_steps=args.max_steps,
                reward=RewardWeights(),
                cell_size=args.cell_size,
                safe_distance=0.6,
                obs_map_size=int(args.obs_map_size),
                terminate_on_collision=False,
            )
            cell_size_m = float(args.cell_size)
        grid = spec.obstacle_grid()

        env_paths: dict[str, PathTrace] = {}
        meta: dict[str, float] = {"cell_size_m": float(cell_size_m)}
        if isinstance(env, AMRBicycleEnv):
            meta["goal_tol_cells"] = float(env.goal_tolerance_m) / float(cell_size_m)
            fp = forest_oriented_box_footprint()
            meta["veh_length_cells"] = float(fp.length) / float(cell_size_m)
            meta["veh_width_cells"] = float(fp.width) / float(cell_size_m)
        else:
            meta["goal_tol_cells"] = 0.5
            meta["veh_length_cells"] = 0.0
            meta["veh_width_cells"] = 0.0
        plot_meta[env_name] = meta

        if not bool(args.skip_rl):
            # Load trained models
            env_obs_dim = int(env.observation_space.shape[0])
            n_actions = int(env.action_space.n)
            agent_cfg = AgentConfig()

            dqn_path = models_dir / env_name / "dqn.pt"
            iddqn_path = models_dir / env_name / "iddqn.pt"
            if not dqn_path.exists() or not iddqn_path.exists():
                raise FileNotFoundError(
                    f"Missing model(s) for env {env_name!r}. Expected: {dqn_path} and {iddqn_path}. "
                    "Point --models at a training run (or an experiment name/dir with a latest run)."
                )

            ckpt_obs_dim = infer_checkpoint_obs_dim(dqn_path)
            if infer_checkpoint_obs_dim(iddqn_path) != ckpt_obs_dim:
                raise RuntimeError(f"Observation dim mismatch between {dqn_path} and {iddqn_path}")

            obs_dim = env_obs_dim
            obs_transform = None
            if ckpt_obs_dim != env_obs_dim:
                raise RuntimeError(
                    f"Checkpoint expects obs_dim={ckpt_obs_dim} but env provides obs_dim={env_obs_dim} for {env_name!r}. "
                    "Re-train models to match the environment observation space."
                )

            dqn_agent = DQNFamilyAgent("dqn", obs_dim, n_actions, config=agent_cfg, seed=args.seed, device=device)
            dqn_agent.load(dqn_path)
            iddqn_agent = DQNFamilyAgent("iddqn", obs_dim, n_actions, config=agent_cfg, seed=args.seed, device=device)
            iddqn_agent.load(iddqn_path)

            # Collect KPI estimates (IDDQN)
            id_kpis: list[KPI] = []
            id_times: list[float] = []
            id_success = 0
            id_trace: PathTrace | None = None
            for i in range(args.runs):
                path, dt, reached = rollout_agent(
                    env,
                    iddqn_agent,
                    max_steps=args.max_steps,
                    seed=args.seed + 10_000 + i,
                    time_mode=str(getattr(args, "kpi_time_mode", "rollout")),
                    obs_transform=obs_transform,
                    forest_adm_horizon=int(args.forest_adm_horizon),
                    forest_topk=int(args.forest_topk),
                    forest_min_od_m=float(args.forest_min_od_m),
                    forest_min_progress_m=float(args.forest_min_progress_m),
                )
                id_times.append(float(dt))
                if bool(reached):
                    id_success += 1
                    raw_corners = float(num_path_corners(path, angle_threshold_deg=13.0))
                    smoothed = smooth_path(path, iterations=2)
                    id_kpis.append(
                        KPI(
                            avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                            num_corners=raw_corners,
                            min_collision_dist=float(min_distance_to_obstacle(grid, smoothed)) * float(cell_size_m),
                            inference_time_s=float(dt),
                            max_corner_deg=float(max_corner_degree(smoothed)),
                        )
                    )
                    if id_trace is None or not bool(id_trace.success):
                        id_trace = PathTrace(path_xy_cells=path, success=True)
                elif id_trace is None:
                    id_trace = PathTrace(path_xy_cells=path, success=False)

            k = mean_kpi(id_kpis)
            k_dict = dict(k.__dict__)
            if id_times:
                k_dict["inference_time_s"] = float(np.mean(id_times))
            rows.append(
                {
                    "Environment": f"Env. ({env_name})",
                    "Algorithm": "IDDQN",
                    "success_rate": float(id_success) / float(max(1, int(args.runs))),
                    **k_dict,
                }
            )
            if id_trace is not None:
                env_paths["IDDQN"] = id_trace

            # DQN
            dq_kpis: list[KPI] = []
            dq_times: list[float] = []
            dq_success = 0
            dq_trace: PathTrace | None = None
            for i in range(args.runs):
                path, dt, reached = rollout_agent(
                    env,
                    dqn_agent,
                    max_steps=args.max_steps,
                    seed=args.seed + 20_000 + i,
                    time_mode=str(getattr(args, "kpi_time_mode", "rollout")),
                    obs_transform=obs_transform,
                    forest_adm_horizon=int(args.forest_adm_horizon),
                    forest_topk=int(args.forest_topk),
                    forest_min_od_m=float(args.forest_min_od_m),
                    forest_min_progress_m=float(args.forest_min_progress_m),
                )
                dq_times.append(float(dt))
                if bool(reached):
                    dq_success += 1
                    raw_corners = float(num_path_corners(path, angle_threshold_deg=13.0))
                    smoothed = smooth_path(path, iterations=2)
                    dq_kpis.append(
                        KPI(
                            avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                            num_corners=raw_corners,
                            min_collision_dist=float(min_distance_to_obstacle(grid, smoothed)) * float(cell_size_m),
                            inference_time_s=float(dt),
                            max_corner_deg=float(max_corner_degree(smoothed)),
                        )
                    )
                    if dq_trace is None or not bool(dq_trace.success):
                        dq_trace = PathTrace(path_xy_cells=path, success=True)
                elif dq_trace is None:
                    dq_trace = PathTrace(path_xy_cells=path, success=False)

            k = mean_kpi(dq_kpis)
            k_dict = dict(k.__dict__)
            if dq_times:
                k_dict["inference_time_s"] = float(np.mean(dq_times))
            rows.append(
                {
                    "Environment": f"Env. ({env_name})",
                    "Algorithm": "DQN",
                    "success_rate": float(dq_success) / float(max(1, int(args.runs))),
                    **k_dict,
                }
            )
            if dq_trace is not None:
                env_paths["DQN"] = dq_trace

        if baselines:
            grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=float(cell_size_m))
            params = default_ackermann_params()
            if env_name in FOREST_ENV_ORDER and isinstance(env, AMRBicycleEnv):
                footprint = forest_oriented_box_footprint()
                goal_xy_tol_m = float(env.goal_tolerance_m)
                goal_theta_tol_rad = float(env.goal_angle_tolerance_rad)
                start_theta_rad = None
            else:
                footprint = point_footprint(cell_size_m=float(cell_size_m))
                goal_xy_tol_m = float(cell_size_m) * 0.5
                goal_theta_tol_rad = float(math.pi)
                start_theta_rad = 0.0

            if "hybrid_astar" in baselines:
                res = plan_hybrid_astar(
                    grid_map=grid_map,
                    footprint=footprint,
                    params=params,
                    start_xy=spec.start_xy,
                    goal_xy=spec.goal_xy,
                    goal_theta_rad=0.0,
                    start_theta_rad=start_theta_rad,
                    goal_xy_tol_m=goal_xy_tol_m,
                    goal_theta_tol_rad=goal_theta_tol_rad,
                    timeout_s=float(args.baseline_timeout),
                    max_nodes=int(args.hybrid_max_nodes),
                )
                ha_plan_ok = bool(res.success)
                ha_exec_path = list(res.path_xy_cells)
                ha_exec_time_s = 0.0
                ha_reached = ha_plan_ok
                if ha_plan_ok and isinstance(env, AMRBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False)):
                    ha_exec_path, ha_exec_time_s, ha_reached = rollout_tracked_path(
                        env,
                        ha_exec_path,
                        max_steps=args.max_steps,
                        seed=args.seed + 30_000,
                    )

                if bool(ha_reached) and ha_exec_path:
                    raw_corners = float(num_path_corners(ha_exec_path, angle_threshold_deg=13.0))
                    smoothed = smooth_path(ha_exec_path, iterations=2)
                    kpi = KPI(
                        avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                        num_corners=raw_corners,
                        min_collision_dist=float(min_distance_to_obstacle(grid, smoothed)) * float(cell_size_m),
                        inference_time_s=float(res.time_s) + float(ha_exec_time_s),
                        max_corner_deg=float(max_corner_degree(smoothed)),
                    )
                else:
                    kpi = mean_kpi([])

                k_dict = dict(kpi.__dict__)
                k_dict["inference_time_s"] = float(res.time_s) + float(ha_exec_time_s)
                rows.append(
                    {
                        "Environment": f"Env. ({env_name})",
                        "Algorithm": "Hybrid A*",
                        "success_rate": 1.0 if bool(ha_reached) else 0.0,
                        **k_dict,
                    }
                )
                env_paths["Hybrid A*"] = PathTrace(path_xy_cells=ha_exec_path, success=bool(ha_reached))

            if "rrt_star" in baselines:
                rrt_kpis: list[KPI] = []
                rrt_times: list[float] = []
                rrt_success = 0
                rrt_trace: PathTrace | None = None
                for i in range(args.runs):
                    res = plan_rrt_star(
                        grid_map=grid_map,
                        footprint=footprint,
                        params=params,
                        start_xy=spec.start_xy,
                        goal_xy=spec.goal_xy,
                        goal_theta_rad=0.0,
                        start_theta_rad=start_theta_rad,
                        goal_xy_tol_m=goal_xy_tol_m,
                        goal_theta_tol_rad=goal_theta_tol_rad,
                        timeout_s=float(args.baseline_timeout),
                        max_iter=int(args.rrt_max_iter),
                        seed=args.seed + 30_000 + i,
                    )
                    exec_path = list(res.path_xy_cells)
                    exec_time_s = 0.0
                    reached = bool(res.success)
                    if bool(res.success) and isinstance(env, AMRBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False)):
                        exec_path, exec_time_s, reached = rollout_tracked_path(
                            env,
                            exec_path,
                            max_steps=args.max_steps,
                            seed=args.seed + 40_000 + i,
                        )

                    rrt_times.append(float(res.time_s) + float(exec_time_s))
                    if bool(reached) and exec_path:
                        rrt_success += 1
                        raw_corners = float(num_path_corners(exec_path, angle_threshold_deg=13.0))
                        smoothed = smooth_path(exec_path, iterations=2)
                        rrt_kpis.append(
                            KPI(
                                avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                                num_corners=raw_corners,
                                min_collision_dist=float(min_distance_to_obstacle(grid, smoothed)) * float(cell_size_m),
                                inference_time_s=float(res.time_s) + float(exec_time_s),
                                max_corner_deg=float(max_corner_degree(smoothed)),
                            )
                        )
                        if rrt_trace is None or not bool(rrt_trace.success):
                            rrt_trace = PathTrace(path_xy_cells=exec_path, success=True)
                    elif rrt_trace is None:
                        rrt_trace = PathTrace(path_xy_cells=exec_path, success=False)

                k = mean_kpi(rrt_kpis)
                k_dict = dict(k.__dict__)
                if rrt_times:
                    k_dict["inference_time_s"] = float(np.mean(rrt_times))
                rows.append(
                    {
                        "Environment": f"Env. ({env_name})",
                        "Algorithm": "RRT*",
                        "success_rate": float(rrt_success) / float(max(1, int(args.runs))),
                        **k_dict,
                    }
                )
                if rrt_trace is not None:
                    env_paths["RRT*"] = rrt_trace

        paths_for_plot[env_name] = env_paths

    table = pd.DataFrame(rows)
    # Pretty column order
    table = table[
        [
            "Environment",
            "Algorithm",
            "success_rate",
            "avg_path_length",
            "num_corners",
            "min_collision_dist",
            "inference_time_s",
            "max_corner_deg",
        ]
    ]
    table = table.copy()

    # Composite metric (lower is better): combines path length and compute time,
    # then penalizes non-reaching behavior via success_rate.
    w_t = float(args.score_time_weight)
    sr_raw = pd.to_numeric(table["success_rate"], errors="coerce").astype(float)
    denom = sr_raw.clip(lower=1e-6)
    base = pd.to_numeric(table["avg_path_length"], errors="coerce").astype(float) + w_t * pd.to_numeric(
        table["inference_time_s"], errors="coerce"
    ).astype(float)
    planning_cost = (base / denom).astype(float)
    planning_cost = planning_cost.where((sr_raw > 0.0) & np.isfinite(base.to_numpy()), other=float("inf"))
    table["planning_cost"] = planning_cost

    table["success_rate"] = pd.to_numeric(table["success_rate"], errors="coerce").astype(float).round(3)
    table["avg_path_length"] = table["avg_path_length"].astype(float).round(3)
    table["num_corners"] = pd.to_numeric(table["num_corners"], errors="coerce").round(0).astype("Int64")
    table["min_collision_dist"] = table["min_collision_dist"].astype(float).round(4)
    table["inference_time_s"] = table["inference_time_s"].astype(float).round(5)
    table["max_corner_deg"] = pd.to_numeric(table["max_corner_deg"], errors="coerce").round(0).astype("Int64")
    table["planning_cost"] = pd.to_numeric(table["planning_cost"], errors="coerce").astype(float).round(3)
    table.to_csv(out_dir / "table2_kpis_raw.csv", index=False)

    table_pretty = table.rename(
        columns={
            "Algorithm": "Algorithm name",
            "success_rate": "Success rate",
            "avg_path_length": "Average path length (m)",
            "num_corners": "Number of path corners",
            "min_collision_dist": "Mini collision to obstacle (m)",
            "inference_time_s": "Inference time (s)",
            "max_corner_deg": "Max corner degree (°)",
            "planning_cost": "Planning cost (m)",
        }
    )
    table_pretty.to_csv(out_dir / "table2_kpis.csv", index=False)
    table_pretty.to_markdown(out_dir / "table2_kpis.md", index=False)

    # Plot Fig. 12-style paths
    envs_to_plot = list(args.envs)[:4]
    n_env = int(len(envs_to_plot))
    cols = 1 if n_env <= 1 else 2
    rows_n = int(math.ceil(float(n_env) / float(cols))) if n_env else 1
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 5.2 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    styles = {
        "IDDQN": dict(color="red", linestyle="-", linewidth=2.0),
        "DQN": dict(color="orangered", linestyle=":", linewidth=2.0),
        "Hybrid A*": dict(color="royalblue", linestyle="--", linewidth=2.0),
        "RRT*": dict(color="seagreen", linestyle="-.", linewidth=2.0),
    }

    for i, env_name in enumerate(envs_to_plot):
        ax = axes[i]
        spec = get_map_spec(env_name)
        grid = spec.obstacle_grid()
        plot_env(ax, grid, title=f"Env. ({env_name})")

        meta = plot_meta.get(env_name, {})

        ax.scatter(
            [spec.start_xy[0]],
            [spec.start_xy[1]],
            marker="^",
            s=80,
            color="blue",
            label="_nolegend_",
        )
        ax.text(spec.start_xy[0] - 1.0, spec.start_xy[1] - 1.0, "SP", fontsize=9, color="black")
        ax.scatter(
            [spec.goal_xy[0]],
            [spec.goal_xy[1]],
            marker="*",
            s=140,
            color="red",
            label="_nolegend_",
        )
        ax.text(spec.goal_xy[0] - 1.0, spec.goal_xy[1] - 1.0, "TP", fontsize=9, color="black")

        tol = float(meta.get("goal_tol_cells", 0.0))
        if tol > 0.0:
            ax.add_patch(
                mpatches.Circle(
                    (float(spec.goal_xy[0]), float(spec.goal_xy[1])),
                    radius=float(tol),
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                )
            )

        env_paths = paths_for_plot.get(env_name, {})
        for algo_name, trace in env_paths.items():
            path = trace.path_xy_cells
            if not path:
                continue
            pts = np.array(path, dtype=np.float32)
            pts_s = chaikin_smooth(pts, iterations=2)
            style = styles.get(algo_name, dict(color="black", linestyle="-", linewidth=1.5))
            label = algo_name if trace.success else f"{algo_name} (fail)"
            alpha = 1.0 if trace.success else 0.55
            ax.plot(pts_s[:, 0], pts_s[:, 1], label=label, alpha=alpha, **style)
            end_marker = "o" if trace.success else "x"
            ax.scatter([float(pts_s[-1, 0])], [float(pts_s[-1, 1])], marker=end_marker, s=28, color=style["color"], label="_nolegend_")

            if float(meta.get("veh_length_cells", 0.0)) > 0.0 and float(meta.get("veh_width_cells", 0.0)) > 0.0:
                draw_vehicle_boxes(
                    ax,
                    trace,
                    length_cells=float(meta["veh_length_cells"]),
                    width_cells=float(meta["veh_width_cells"]),
                    color=str(style["color"]),
                )

        ax.legend(fontsize=8, loc="lower right")

    for ax in axes[n_env:]:
        ax.axis("off")

    fig.suptitle("Simulation results of different path-planning methods")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig12_paths.png", dpi=200)
    plt.close(fig)

    print(f"Wrote: {out_dir / 'fig12_paths.png'}")
    print(f"Wrote: {out_dir / 'table2_kpis.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_raw.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis.md'}")
    print(f"Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
