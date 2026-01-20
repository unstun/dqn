from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from amr_dqn.runtime import configure_runtime, select_device, torch_runtime_info
from amr_dqn.runs import create_run_dir, resolve_experiment_dir

configure_runtime()

import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
import torch

from amr_dqn.agents import AgentConfig, DQNFamilyAgent
from amr_dqn.env import AMRBicycleEnv, AMRGridEnv, RewardWeights
from amr_dqn.maps import ENV_ORDER, FOREST_ENV_ORDER, get_map_spec


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = int(window)
    if x.size < w:
        return x
    kernel = np.ones((w,), dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same")


def forest_demo_target(*, learning_starts: int, batch_size: int) -> int:
    # Forest long-horizon runs can fall into a "stop until stuck" local optimum unless the replay
    # is initially dominated by successful expert trajectories. Empirically, forest_a needs ~20k
    # demo transitions (the cap) for stable imitation + TD bootstrapping.
    target = max(int(learning_starts) * 40, int(batch_size))
    return int(min(int(target), 20_000))


def forest_expert_action(
    env: AMRBicycleEnv,
    *,
    forest_expert: str,
    horizon_steps: int,
    w_clearance: float,
) -> int:
    h = max(1, int(horizon_steps))
    if str(forest_expert).lower() == "hybrid_astar":
        # Safer Hybrid A* tracking for demonstrations / guided exploration.
        # The shorter-horizon aggressive tracker can collide on harder maps (notably forest_a).
        return env.expert_action_hybrid_astar(
            lookahead_points=5,
            horizon_steps=max(15, h),
            w_target=0.2,
            w_heading=0.2,
            w_clearance=float(w_clearance),
            w_speed=0.0,
        )
    return env.expert_action_mpc(
        horizon_steps=max(15, h),
        w_clearance=float(w_clearance),
    )


def collect_forest_demos(
    env: AMRBicycleEnv,
    *,
    target: int,
    seed: int,
    forest_curriculum: bool,
    curriculum_band_m: float,
    forest_expert: str,
    forest_demo_horizon: int,
    forest_demo_w_clearance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_dim = int(env.observation_space.shape[0])
    n = max(0, int(target))
    obs_buf = np.zeros((n, obs_dim), dtype=np.float32)
    next_obs_buf = np.zeros((n, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n,), dtype=np.int64)
    rew_buf = np.zeros((n,), dtype=np.float32)
    done_buf = np.zeros((n,), dtype=np.float32)

    added = 0
    demo_ep = 0
    demo_prog = np.linspace(0.0, 1.0, num=5, dtype=np.float32)
    while added < n and demo_ep < 200:
        opts = None
        if forest_curriculum:
            p = float(demo_prog[demo_ep % int(demo_prog.size)])
            opts = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
        obs, _ = env.reset(seed=int(seed) + 50_000 + int(demo_ep), options=opts)
        done = False
        truncated = False
        while not (done or truncated) and added < n:
            a = forest_expert_action(
                env,
                forest_expert=str(forest_expert),
                horizon_steps=int(forest_demo_horizon),
                w_clearance=float(forest_demo_w_clearance),
            )
            next_obs, reward, done, truncated, _info = env.step(int(a))
            obs_buf[added] = obs
            next_obs_buf[added] = next_obs
            act_buf[added] = int(a)
            rew_buf[added] = float(reward)
            done_buf[added] = 1.0 if bool(done) else 0.0
            obs = next_obs
            added += 1
        demo_ep += 1

    return (
        obs_buf[:added],
        act_buf[:added],
        rew_buf[:added],
        next_obs_buf[:added],
        done_buf[:added],
    )


def train_one(
    env: gym.Env,
    algo: str,
    *,
    episodes: int,
    seed: int,
    out_dir: Path,
    agent_cfg: AgentConfig,
    train_freq: int,
    learning_starts: int,
    forest_curriculum: bool,
    curriculum_band_m: float,
    curriculum_ramp: float,
    forest_demo_prefill: bool,
    forest_demo_pretrain_steps: int,
    forest_demo_horizon: int,
    forest_demo_w_clearance: float,
    forest_demo_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    forest_expert: str,
    forest_expert_exploration: bool,
    forest_expert_prob_start: float,
    forest_expert_prob_final: float,
    forest_expert_prob_decay: float,
    progress: bool,
    device: torch.device,
) -> tuple[DQNFamilyAgent, np.ndarray]:
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    agent = DQNFamilyAgent(algo, obs_dim, n_actions, config=agent_cfg, seed=seed, device=device)

    returns = np.zeros((episodes,), dtype=np.float32)
    global_step = 0

    best_score: tuple[int, int, int] = (-1, -10**18, 0)
    best_q: dict[str, torch.Tensor] | None = None
    best_q_target: dict[str, torch.Tensor] | None = None
    best_train_steps: int = 0
    pretrain_q: dict[str, torch.Tensor] | None = None
    pretrain_q_target: dict[str, torch.Tensor] | None = None
    pretrain_train_steps: int = 0
    explore_rng = np.random.default_rng(seed + 777)

    def episode_score(*, reached: bool, collision: bool, steps: int, ret: float) -> tuple[int, int, int]:
        """Prefer reach > survive (timeout) > collision (then higher return)."""
        if bool(reached):
            # Prefer higher return, then fewer steps.
            return (2, int(1_000_000 * float(ret)), -int(steps))
        if bool(collision):
            return (0, int(1_000_000 * float(ret)), -int(steps))
        # Timeout / did not reach.
        return (1, int(1_000_000 * float(ret)), -int(steps))

    def clone_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def forest_expert_action_local() -> int:
        if not isinstance(env, AMRBicycleEnv):
            raise RuntimeError("forest_expert_action called for non-forest env")
        return forest_expert_action(
            env,
            forest_expert=str(forest_expert),
            horizon_steps=int(forest_demo_horizon),
            w_clearance=float(forest_demo_w_clearance),
        )

    # Forest-only: prefill replay buffer with a few short expert rollouts.
    # This is off-policy data (valid for Q-learning) and dramatically reduces the
    # chance of converging to the degenerate "stop until stuck" behavior.
    if forest_demo_prefill and isinstance(env, AMRBicycleEnv) and learning_starts > 0:
        demo_target = forest_demo_target(learning_starts=int(learning_starts), batch_size=int(agent_cfg.batch_size))
        if forest_demo_data is not None:
            obs_buf, act_buf, rew_buf, next_obs_buf, done_buf = forest_demo_data
            n = int(min(int(demo_target), int(obs_buf.shape[0])))
            for i in range(n):
                agent.observe(
                    obs_buf[i],
                    int(act_buf[i]),
                    float(rew_buf[i]),
                    next_obs_buf[i],
                    bool(done_buf[i] > 0.5),
                    demo=True,
                )
            global_step += int(n)
        else:
            # Collect *more* than just learning_starts transitions: imitation on static global maps
            # is very data-efficient, and preserving a diverse demo set improves robustness when the
            # learned policy slightly deviates from the reference trajectory (DAgger-like effect).
            demo_added = 0
            demo_ep = 0
            demo_prog = np.linspace(0.0, 1.0, num=5, dtype=np.float32)
            while demo_added < demo_target and demo_ep < 200:
                opts = None
                if forest_curriculum:
                    # When curriculum is enabled, diversify demonstration starts to match the
                    # training start-state distribution.
                    p = float(demo_prog[demo_ep % int(demo_prog.size)])
                    opts = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
                obs, _ = env.reset(seed=seed + 50_000 + demo_ep, options=opts)
                done = False
                truncated = False
                while not (done or truncated) and demo_added < demo_target:
                    a = forest_expert_action_local()
                    next_obs, reward, done, truncated, _info = env.step(a)
                    agent.observe(obs, a, reward, next_obs, done, demo=True)
                    obs = next_obs
                    demo_added += 1
                    global_step += 1
                demo_ep += 1

        # Supervised warm-start on demos before TD learning.
        pre_steps = int(max(0, int(forest_demo_pretrain_steps)))
        if pre_steps > 0:
            # Run in chunks and stop early once the greedy (masked) policy can reach the goal.
            done_steps = 0
            chunk = 2000
            while done_steps < pre_steps:
                n = min(int(chunk), int(pre_steps - done_steps))
                agent.pretrain_on_demos(steps=int(n))
                done_steps += int(n)

                # Quick self-check: use the same admissible-action mask as inference.
                obs_eval, _ = env.reset(seed=seed + 99_999)
                done_eval = False
                trunc_eval = False
                reached_eval = False
                while not (done_eval or trunc_eval):
                    a_eval = agent.act(obs_eval, episode=0, explore=False)
                    if not bool(env.is_action_admissible(int(a_eval), horizon_steps=15, min_od_m=0.0, min_progress_m=1e-4)):
                        mask = env.admissible_action_mask(horizon_steps=15, min_od_m=0.0, min_progress_m=1e-4)
                        if bool(mask.any()):
                            a_eval = agent.act_masked(obs_eval, episode=0, explore=False, action_mask=mask)
                    obs_eval, _r, done_eval, trunc_eval, info_eval = env.step(int(a_eval))
                    if bool(info_eval.get("reached", False)):
                        reached_eval = True
                        break
                if reached_eval:
                    break

            # Sync target net after imitation so subsequent TD updates start from a consistent pair.
            agent.q_target.load_state_dict(agent.q.state_dict())

            # Keep a snapshot of the post-imitation policy: it is often the most reliable
            # goal-reaching policy on static maps, while later TD updates can sometimes drift.
            pretrain_q = clone_state_dict(agent.q.state_dict())
            pretrain_q_target = clone_state_dict(agent.q_target.state_dict())
            pretrain_train_steps = int(agent._train_steps)

        # Start learning immediately once the buffer has useful transitions.
        global_step = max(int(global_step), int(learning_starts))

    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            pbar = None
        else:
            pbar = tqdm(
                range(episodes),
                desc=f"Train {env.map_spec.name} {algo}",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
            )

    ep_iter = pbar if pbar is not None else range(episodes)
    for ep in ep_iter:
        reset_options = None
        if forest_curriculum and isinstance(env, AMRBicycleEnv):
            p_raw = float(ep) / float(max(1, episodes - 1))
            ramp = max(1e-6, float(curriculum_ramp))
            p = float(np.clip(p_raw / ramp, 0.0, 1.0))
            reset_options = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
        obs, _ = env.reset(seed=seed + ep, options=reset_options)
        ep_return = 0.0
        done = False
        truncated = False
        ep_steps = 0
        last_info: dict[str, object] = {}

        while not (done or truncated):
            ep_steps += 1
            global_step += 1
            # Forest stabilizer: mix in an expert as the *behavior* policy early in training.
            # Off-policy Q-learning remains valid, while successful trajectories become frequent
            # enough for bootstrapping long-horizon returns.
            used_expert = False
            if forest_expert_exploration and isinstance(env, AMRBicycleEnv):
                ramp = max(1e-6, float(forest_expert_prob_decay))
                t = float(np.clip((float(ep) / float(max(1, episodes - 1))) / ramp, 0.0, 1.0))
                p_exp = float(forest_expert_prob_start) + (float(forest_expert_prob_final) - float(forest_expert_prob_start)) * t
                p_exp = float(np.clip(p_exp, 0.0, 1.0))
                if explore_rng.random() < p_exp:
                    action = forest_expert_action_local()
                    used_expert = True
                else:
                    action = agent.act(obs, episode=ep, explore=True)
                    if not bool(env.is_action_admissible(int(action), horizon_steps=6, min_od_m=0.0, min_progress_m=1e-4)):
                        mask = env.admissible_action_mask(horizon_steps=6, min_od_m=0.0, min_progress_m=1e-4)
                        action = agent.act_masked(obs, episode=ep, explore=True, action_mask=mask)
            else:
                action = agent.act(obs, episode=ep, explore=True)
            next_obs, reward, done, truncated, info = env.step(action)
            last_info = dict(info)
            # Time-limit truncation should not be treated as terminal for bootstrapping.
            agent.observe(obs, action, reward, next_obs, done, demo=used_expert)
            ep_return += float(reward)

            if global_step >= learning_starts and (global_step % max(1, train_freq) == 0):
                agent.update()

            obs = next_obs

        returns[ep] = float(ep_return)

        if pbar is not None:
            pbar.set_postfix(
                {
                    "ret": f"{ep_return:.3f}",
                    "eps": f"{agent.epsilon(ep):.3f}",
                    "steps": global_step,
                    "updates": int(agent._train_steps),
                },
                refresh=False,
            )

        reached = bool(last_info.get("reached", False))
        collision = bool(last_info.get("collision", False) or last_info.get("stuck", False))
        score = episode_score(reached=reached, collision=collision, steps=ep_steps, ret=ep_return)
        if score > best_score:
            best_score = score
            best_q = clone_state_dict(agent.q.state_dict())
            best_q_target = clone_state_dict(agent.q_target.state_dict())
            best_train_steps = int(agent._train_steps)

    final_q = clone_state_dict(agent.q.state_dict())
    final_q_target = clone_state_dict(agent.q_target.state_dict())
    final_train_steps = int(agent._train_steps)

    def eval_greedy(q_sd: dict[str, torch.Tensor], q_target_sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
        agent.q.load_state_dict(q_sd)
        agent.q_target.load_state_dict(q_target_sd)

        obs, _ = env.reset(seed=seed + 9999)
        done = False
        truncated = False
        steps = 0
        ret = 0.0
        last_info: dict[str, object] = {}
        while not (done or truncated):
            steps += 1
            if isinstance(env, AMRBicycleEnv):
                a = agent.act(obs, episode=0, explore=False)
                if not bool(env.is_action_admissible(int(a), horizon_steps=15, min_od_m=0.0, min_progress_m=1e-4)):
                    mask = env.admissible_action_mask(horizon_steps=15, min_od_m=0.0, min_progress_m=1e-4)
                    if bool(mask.any()):
                        a = agent.act_masked(obs, episode=0, explore=False, action_mask=mask)
            else:
                a = agent.act(obs, episode=0, explore=False)
            obs, r, done, truncated, info = env.step(a)
            last_info = dict(info)
            ret += float(r)

        reached = bool(last_info.get("reached", False))
        collision = bool(last_info.get("collision", False) or last_info.get("stuck", False))
        return episode_score(reached=reached, collision=collision, steps=steps, ret=ret)

    # Choose between the final policy and the best (exploratory) episode checkpoint based on greedy performance.
    best_greedy_score = eval_greedy(final_q, final_q_target)
    chosen_q, chosen_q_target, chosen_train_steps = final_q, final_q_target, final_train_steps

    if best_q is not None and best_q_target is not None:
        candidate_score = eval_greedy(best_q, best_q_target)
        if candidate_score > best_greedy_score:
            best_greedy_score = candidate_score
            chosen_q, chosen_q_target, chosen_train_steps = best_q, best_q_target, best_train_steps

    if pretrain_q is not None and pretrain_q_target is not None:
        candidate_score = eval_greedy(pretrain_q, pretrain_q_target)
        if candidate_score > best_greedy_score:
            best_greedy_score = candidate_score
            chosen_q, chosen_q_target, chosen_train_steps = pretrain_q, pretrain_q_target, pretrain_train_steps

    agent.q.load_state_dict(chosen_q)
    agent.q_target.load_state_dict(chosen_q_target)
    agent._train_steps = int(chosen_train_steps)

    model_path = out_dir / "models" / env.map_spec.name / f"{algo}.pt"
    agent.save(model_path)
    return agent, returns


def main() -> int:
    ap = argparse.ArgumentParser(description="Train DQN and IDDQN and generate Fig. 13-style plot.")
    ap.add_argument(
        "--envs",
        nargs="*",
        default=list(ENV_ORDER),
        help="Subset of envs: a b c d forest_a forest_b forest_c forest_d",
    )
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--sensor-range", type=int, default=6)
    ap.add_argument(
        "--n-sectors",
        type=int,
        default=36,
        help="Forest lidar sectors (36=10°, 72=5°). Ignored for non-forest envs.",
    )
    ap.add_argument("--cell-size", type=float, default=1.0, help="Grid cell size in meters.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Experiment name/dir (bare names are stored under --runs-root).",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="If --out is a bare name, store it under this directory.",
    )
    ap.add_argument(
        "--timestamp-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write into <experiment>/train_<timestamp>/ to avoid mixing outputs.",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device selection (default: auto).",
    )
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (when using --device=cuda).")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Print CUDA/runtime info and exit (use to verify CUDA setup).",
    )
    ap.add_argument("--train-freq", type=int, default=4)
    ap.add_argument("--learning-starts", type=int, default=500)
    ap.add_argument("--ma-window", type=int, default=20, help="Moving average window for plotting (1=raw).")
    ap.add_argument(
        "--obs-map-size",
        type=int,
        default=12,
        help="Downsampled global-map observation size (applies to both grid and forest envs).",
    )
    ap.add_argument(
        "--forest-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only curriculum: start closer to the goal, then expand to the full start-goal distance.",
    )
    ap.add_argument(
        "--curriculum-band-m",
        type=float,
        default=2.0,
        help="Curriculum band width (meters) for sampling start states by cost-to-go.",
    )
    ap.add_argument(
        "--curriculum-ramp",
        type=float,
        default=0.35,
        help=(
            "Forest curriculum ramp fraction (0<r<=1). The fixed-start probability and curriculum distance "
            "reach 1.0 by r*episodes (smaller = harder sooner, but avoids train/test mismatch)."
        ),
    )
    ap.add_argument(
        "--forest-demo-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only: prefill replay with expert rollouts (stabilizes training).",
    )
    ap.add_argument(
        "--forest-demo-pretrain-steps",
        type=int,
        default=50_000,
        help="Forest-only: supervised warm-start steps on demo transitions (behavior cloning + margin).",
    )
    ap.add_argument(
        "--forest-demo-horizon",
        type=int,
        default=15,
        help="Forest-only: expert horizon steps for demo prefill (constant action).",
    )
    ap.add_argument(
        "--forest-demo-w-clearance",
        type=float,
        default=0.8,
        help="Forest-only: expert clearance weight for demo prefill.",
    )
    ap.add_argument(
        "--forest-expert-exploration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only: mix expert actions into the behavior policy (stabilizes long-horizon learning).",
    )
    ap.add_argument(
        "--forest-expert",
        choices=("hybrid_astar", "mpc"),
        default="hybrid_astar",
        help="Forest-only: expert source used for demos / guided exploration (default: hybrid_astar).",
    )
    ap.add_argument(
        "--forest-expert-prob-start",
        type=float,
        default=0.7,
        help="Forest-only: probability of using expert instead of the agent's epsilon-greedy action selection (start).",
    )
    ap.add_argument(
        "--forest-expert-prob-final",
        type=float,
        default=0.0,
        help="Forest-only: probability of using expert instead of the agent's epsilon-greedy action selection (final).",
    )
    ap.add_argument(
        "--forest-expert-prob-decay",
        type=float,
        default=0.6,
        help="Forest-only: decay fraction (0<d<=1) for expert exploration probability.",
    )
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a training progress bar (default: on when running in a TTY).",
    )
    args = ap.parse_args()

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

    progress = bool(sys.stderr.isatty()) if args.progress is None else bool(args.progress)

    experiment_dir = resolve_experiment_dir(args.out, runs_root=args.runs_root)
    run_paths = create_run_dir(experiment_dir, timestamp_runs=args.timestamp_runs, prefix="train")
    out_dir = run_paths.run_dir

    agent_cfg = AgentConfig()
    dqn_cfg = replace(agent_cfg, eps_start=0.6)
    iddqn_cfg = replace(agent_cfg, eps_start=0.6, per_alpha=0.2, target_tau=0.005)
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
                "kind": "train",
                "argv": list(sys.argv),
                "experiment_dir": str(run_paths.experiment_dir),
                "run_dir": str(run_paths.run_dir),
                "args": args_payload,
                "torch": asdict(torch_runtime_info()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "configs" / "agent_config.json").write_text(json.dumps(asdict(agent_cfg), indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "configs" / "agent_config_dqn.json").write_text(json.dumps(asdict(dqn_cfg), indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "configs" / "agent_config_iddqn.json").write_text(json.dumps(asdict(iddqn_cfg), indent=2, sort_keys=True), encoding="utf-8")

    all_rows: list[dict[str, float | int | str]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}

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
            forest_demo_data = None
            if bool(args.forest_demo_prefill) and int(args.learning_starts) > 0:
                demo_target = forest_demo_target(learning_starts=int(args.learning_starts), batch_size=int(agent_cfg.batch_size))
                forest_demo_data = collect_forest_demos(
                    env,
                    target=int(demo_target),
                    seed=int(args.seed + 1000),
                    forest_curriculum=bool(args.forest_curriculum),
                    curriculum_band_m=float(args.curriculum_band_m),
                    forest_expert=str(args.forest_expert),
                    forest_demo_horizon=int(args.forest_demo_horizon),
                    forest_demo_w_clearance=float(args.forest_demo_w_clearance),
                )
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
            forest_demo_data = None

        _, dqn_returns = train_one(
            env,
            "dqn",
            episodes=args.episodes,
            seed=args.seed + 1000,
            out_dir=out_dir,
            agent_cfg=dqn_cfg,
            train_freq=args.train_freq,
            learning_starts=args.learning_starts,
            forest_curriculum=bool(args.forest_curriculum),
            curriculum_band_m=float(args.curriculum_band_m),
            curriculum_ramp=float(args.curriculum_ramp),
            forest_demo_prefill=bool(args.forest_demo_prefill),
            forest_demo_pretrain_steps=int(args.forest_demo_pretrain_steps),
            forest_demo_horizon=int(args.forest_demo_horizon),
            forest_demo_w_clearance=float(args.forest_demo_w_clearance),
            forest_demo_data=forest_demo_data,
            forest_expert=str(args.forest_expert),
            forest_expert_exploration=bool(args.forest_expert_exploration),
            forest_expert_prob_start=float(args.forest_expert_prob_start),
            forest_expert_prob_final=float(args.forest_expert_prob_final),
            forest_expert_prob_decay=float(args.forest_expert_prob_decay),
            progress=progress,
            device=device,
        )
        _, iddqn_returns = train_one(
            env,
            "iddqn",
            episodes=args.episodes,
            # Forest training (global-map + imitation warm-start) is currently sensitive to
            # random initialization. Use the same deterministic seed offset as DQN so both
            # models reliably converge to goal-reaching policies on the fixed forest maps.
            seed=args.seed + 1000,
            out_dir=out_dir,
            agent_cfg=iddqn_cfg,
            train_freq=args.train_freq,
            learning_starts=args.learning_starts,
            forest_curriculum=bool(args.forest_curriculum),
            curriculum_band_m=float(args.curriculum_band_m),
            curriculum_ramp=float(args.curriculum_ramp),
            forest_demo_prefill=bool(args.forest_demo_prefill),
            forest_demo_pretrain_steps=int(args.forest_demo_pretrain_steps),
            forest_demo_horizon=int(args.forest_demo_horizon),
            forest_demo_w_clearance=float(args.forest_demo_w_clearance),
            forest_demo_data=forest_demo_data,
            forest_expert=str(args.forest_expert),
            forest_expert_exploration=bool(args.forest_expert_exploration),
            forest_expert_prob_start=float(args.forest_expert_prob_start),
            forest_expert_prob_final=float(args.forest_expert_prob_final),
            forest_expert_prob_decay=float(args.forest_expert_prob_decay),
            progress=progress,
            device=device,
        )

        curves[env_name] = {"dqn": dqn_returns, "iddqn": iddqn_returns}
        for ep in range(args.episodes):
            all_rows.append(
                {
                    "env": env_name,
                    "episode": ep + 1,
                    "dqn_return": float(dqn_returns[ep]),
                    "iddqn_return": float(iddqn_returns[ep]),
                }
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "training_returns.csv", index=False)

    # Plot Fig. 13-style reward curves
    envs_to_plot = list(args.envs)[:4]
    n_env = int(len(envs_to_plot))
    cols = 1 if n_env <= 1 else 2
    rows_n = int(np.ceil(float(n_env) / float(cols))) if n_env else 1
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 3.8 * rows_n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    used = 0
    for i, env_name in enumerate(envs_to_plot):
        ax = axes[i]
        dqn = curves[env_name]["dqn"]
        iddqn = curves[env_name]["iddqn"]
        dqn_plot = moving_average(dqn, args.ma_window)
        iddqn_plot = moving_average(iddqn, args.ma_window)

        ax.plot(range(1, args.episodes + 1), iddqn_plot, label="IDDQN", linewidth=1.0)
        ax.plot(range(1, args.episodes + 1), dqn_plot, label="DQN", linewidth=1.0)
        ax.set_title(f"Env. ({env_name})")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Rewards")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        used += 1

    for ax in axes[n_env:]:
        ax.axis("off")

    fig.suptitle("Changes in the reward values of IDDQN and DQN during the training process")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig13_rewards.png", dpi=200)
    plt.close(fig)

    print(f"Wrote: {out_dir / 'fig13_rewards.png'}")
    print(f"Wrote: {out_dir / 'training_returns.csv'}")
    print(f"Wrote models under: {out_dir / 'models'}")
    print(f"Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
