# Forest (Bicycle) + (I)DDQN: Global-Planning Implementation Notes

This repository is a **global planning** codebase: the full occupancy grid map is known. For forest scenarios we simulate Ackermann/bicycle kinematics on a 0.1 m occupancy grid and train (I)DDQN on a **map-based observation** (not sensor-only). A Hybrid A* planner is used as the DQfD-style expert for demonstrations and optional guided exploration.

Key code locations:

- Environment (forest bicycle): `amr_dqn/env.py` (`AMRBicycleEnv`)
- Forest map generation: `amr_dqn/maps/forest.py`, `amr_dqn/maps/__init__.py`
- Agent + losses: `amr_dqn/agents.py`, networks in `amr_dqn/networks.py`
- Training / inference CLIs: `amr_dqn/cli/train.py`, `amr_dqn/cli/infer.py`
- Baseline planners (Hybrid A*, RRT*): `amr_dqn/baselines/pathplan.py`

## 1) Map model (0.1 m occupancy grid)

Forest maps are binary grids (`0=free`, `1=occupied`) with `y=0` at the bottom (consistent with plotting).

To compute continuous collision clearance we precompute an EDT distance field `D(x)` (meters) to the nearest obstacle cell:

```
D(x) = EDT(occupancy)(x) * cell_size
```

World boundaries are treated as obstacles by clamping the EDT with the distance-to-boundary. See `AMRBicycleEnv.__init__`.

## 2) Bicycle dynamics

Each step integrates a rear-axle bicycle model with discrete actions `(δ̇, a)`:

```
v_{k+1}   = clip(v_k + a Δt, 0, v_max)
δ_{k+1}   = clip(δ_k + δ̇ Δt, -δ_lim(v_{k+1}), +δ_lim(v_{k+1}))
x_{k+1}   = x_k + v_{k+1} cos(ψ_k) Δt
y_{k+1}   = y_k + v_{k+1} sin(ψ_k) Δt
ψ_{k+1}   = wrap(ψ_k + (v_{k+1}/L) tan(δ_{k+1}) Δt)
```

Steering is speed-limited via a yaw-rate constraint:

```
δ_lim(v) = atan( min( tan(δ_max), (ω_max L)/max(v, ε) ) )
```

See `bicycle_integrate_one_step(...)` and `steering_limit_rad(...)` in `amr_dqn/env.py`.

## 3) Collision model: two-circle footprint + clearance (OD)

We approximate the vehicle by two circles, sampled on the EDT with bilinear interpolation.

- Nominal dimensions: length `l=0.924 m`, width `w=0.740 m`
- Circle radius:
  ```
  r = sqrt( (w/2)^2 + (l/4)^2 ) ≈ 0.436 m
  ```

Clearance to obstacles (OD) is defined as:

```
OD(x,ψ) = min_i ( D(c_i(x,ψ)) - r )
```

- `OD > 0`: collision-free clearance
- `OD ≤ 0`: penetration (collision)

In plotting and Hybrid A*, an oriented box footprint consistent with the same dimensions is used:
`length=0.924 m`, `width=0.740 m` (`forest_oriented_box_footprint()`).

## 4) Observation (global planning, map-known)

Forest environments use a **global-map observation** (no sensor-only lidar state). The observation is:

1) **Scalars** (normalized to `[-1,1]`): agent `(x,y)`, goal `(x_g,y_g)`, `sin(ψ)`, `cos(ψ)`, `v`, `δ`, normalized cost-to-go, relative goal angle `α`, and clearance `OD`.
2) **Static maps** (downsampled to `obs_map_size × obs_map_size` and flattened):
   - occupancy grid
   - cost-to-go field

So:

```
obs_dim = 11 + 2 * obs_map_size^2
```

See `AMRBicycleEnv._observe()` in `amr_dqn/env.py`.

## 5) Reward (goal-reaching + forest-safe driving)

The reward is dense and uses clearance-aware progress:

```
r = k_p (C(s_k) - C(s_{k+1}))  - k_t  - k_δ (δ_{k+1}-δ_k)^2  - k_a (a_k-a_{k-1})^2 (v/v_max)^2  - k_κ tan^2(δ_{k+1})  + safety  + terminal
```

Safety terms (using `OD`):

- Near-obstacle penalty when `OD < d_safe`:
  ```
  r_obs = -k_o ( 1/(OD+ε) - 1/(d_safe+ε) )
  ```
- Near-obstacle speed coupling (discourages fast driving in tight corridors):
  ```
  r_speed = -k_v ((d_safe - OD)/d_safe) (v/v_max)^2
  ```

Terminal:

- collision: `-200` (terminate)
- reached goal: `+200` (terminate when position is within tolerance and heading is within tolerance)

Default positional goal tolerance is **0.50 m** (`goal_tolerance_m=0.50`) to avoid brittle failures with discrete controls.

## 6) Discrete action set (35 actions)

Action space is the Cartesian product of 7 steering-rate values and 5 accelerations:

- `δ̇ ∈ { -δ̇_max, -2/3 δ̇_max, -1/3 δ̇_max, 0, +1/3 δ̇_max, +2/3 δ̇_max, +δ̇_max }`
- `a ∈ { -a_max, -0.5 a_max, 0, +0.5 a_max, +a_max }`

See `build_ackermann_action_table_35(...)`.

## 7) Cost-to-go field (progress shaping + masks)

We compute an 8-connected Dijkstra cost-to-go field `C_cell(x)` on a *clearance-feasible* traversability mask:

```
traversable(x) = ( D(x) > r + ε_cell )
```

The continuous pose cost used for shaping and masking samples `C_cell` under bilinear interpolation, with a finite fill value to prevent `inf` bleeding into neighbors:

- `bilinear_sample_2d_finite(...)` in `amr_dqn/env.py`

## 8) Curriculum (forest-only)

When enabled, early training episodes start closer to the goal and gradually expand back to the canonical start.

Important implementation detail: curriculum start states are preferentially sampled **along the precomputed Hybrid A* reference path**, so the expert remains available and does not repeatedly time out re-planning from many random starts. See `AMRBicycleEnv.reset(..., options=...)`.

## 9) Hybrid A* expert (DQfD demonstrations)

The DQfD-style expert is Hybrid A* based:

1) A deterministic Hybrid A* path is precomputed for each forest map and stored under:
   - `amr_dqn/maps/precomputed/forest_*_hybrid_astar_path.json`
2) At runtime, `AMRBicycleEnv._hybrid_astar_path()` loads/caches that path (canonical start).
3) `AMRBicycleEnv.expert_action_hybrid_astar(...)` tracks the path with a lookahead waypoint and picks the best discrete action over a short rollout horizon, balancing:
   - global progress (cost-to-go)
   - staying close to the reference path
   - clearance (OD)

Important detail (robustness): tracking parameters matter. A too-short horizon with overly aggressive waypoint/heading chasing can collide on harder maps (notably `forest_a`). The training CLI uses a safer Hybrid A* expert configuration (longer horizon, larger lookahead, higher clearance emphasis) for both demo prefill and guided exploration.

If Hybrid A* is unavailable for a given start (e.g., planning fails), the expert falls back to a simple MPC-style one-step search (`expert_action_mpc`).

## 10) Stabilizers used to prevent “never reach goal”

Forest long-horizon training is prone to degenerate policies (stop/jitter). The default forest training CLI enables stabilizers that keep learning in the goal-reaching regime:

- Demo prefill: collect expert rollouts into replay before TD learning begins (`--forest-demo-prefill`).
- Demo volume: prefill is scaled up (capped at 20k transitions) so the replay starts with enough successful trajectories on harder maps.
- Demo reuse: in a single training run, the demo rollout is collected once per environment and shared between DQN and IDDQN (saves time).
- Demo preservation: replay buffers avoid overwriting expert transitions with non-demo transitions (`amr_dqn/replay_buffer.py`).
- Supervised warm start: behavior cloning + large-margin loss on demo transitions (`--forest-demo-pretrain-steps`, `DQNFamilyAgent.pretrain_on_demos`).
- Guided exploration: mix expert actions into the behavior policy early in training (`--forest-expert-exploration` + `--forest-expert-prob-*`).
- IDDQN: uses prioritized replay + soft target updates by default (vs. DQN), improving stability on long-horizon forests.
- Safety/admissible action masks during training and inference:
  - `safe_action_mask(...)`: collision-safe short-horizon actions
  - `admissible_action_mask(...)`: collision-safe + cost-to-go progress

Inference does **not** “gate” actions with the teacher; it evaluates the learned greedy policy under an admissible-action mask for safety/robustness.

For performance, inference avoids computing the full admissible-action mask on every step: it first checks whether the greedy action is admissible (`AMRBicycleEnv.is_action_admissible`), then tries a small top-k fallback, and only computes the full mask as a last resort.

## 11) Evaluation outputs (KPIs + plotting)

`infer.py` reports per-environment KPIs and writes:

- `table2_kpis_raw.csv`, `table2_kpis.csv`, `table2_kpis.md`
- `fig12_paths.png`: overlays DQN, IDDQN, Hybrid A*, RRT* paths

Plot improvements implemented:

- Start/goal markers + goal tolerance circle
- End markers and "(fail)" labels when a method does not reach the goal
- Vehicle oriented boxes drawn along trajectories (visual footprint)
- Tick formatting to avoid overlapping axis labels

Recommended comparison metric (lower is better):

```
planning_cost = (avg_path_length + w * inference_time_s) / max(success_rate, eps)
```

- `w` is `--score-time-weight` (default: `0.5` m/s)
- `eps` is a small constant to avoid divide-by-zero

Notes on `inference_time_s`:

- Default (`--kpi-time-mode rollout`) measures *full rollout wall time* (includes `env.step`).
- `--kpi-time-mode policy` measures only *action-selection compute time* (Q forward + admissibility checks), which is typically the fairest comparison vs planner runtimes.
- Forest-only: `--forest-baseline-rollout` rolls out a discrete tracker on the planned baseline path and adds the tracker compute to `inference_time_s` (planning + tracking).

## 12) Recommended commands

Train a single forest map (fixed start; DQfD-style stabilizers on):

```
python train.py --envs forest_b --out outputs_forest_dqfd3 --episodes 600 --max-steps 600 --device auto --no-timestamp-runs --no-forest-curriculum --forest-expert-prob-start 1.0 --forest-expert-prob-final 0.0 --forest-expert-prob-decay 0.7
```

Inference with baselines:

```
python infer.py --envs forest_b --models outputs_forest_dqfd3 --out outputs_forest_dqfd3_infer --baselines all --baseline-timeout 5 --hybrid-max-nodes 200000 --rrt-max-iter 5000 --kpi-time-mode policy --device auto
```
