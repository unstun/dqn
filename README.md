# dqn (forest vehicle planning + tracking)

English | [简体中文](README.zh-CN.md)

This repo focuses on a forest-scene kinematic vehicle (Ackermann/bicycle) environment:
`forest_a`, `forest_b`, `forest_c`, `forest_d`.

Default conda environment: `ros2py310`.

## AI TL;DR (contract, 2026-02-21)

```text
STABLE_PROFILE=v7p1
CLAIM_REGIME=shielded/hybrid (do not label as strict-argmax)
SMOKE_GATE=train episodes=150; infer runs=3 (screening only; no final claims)
FINAL_GATE=short runs=20 + long runs=20; pass iff:
  - success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)
  - avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)
  - path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)
REPORT_ARTIFACTS:
  - table2_kpis_mean_raw.csv (mean KPIs; use for gates/tables)
  - table2_kpis_raw.csv (per-run/per-pair diagnostics; includes failure_reason)
```

## Stages (terminology)

- `self-check`: imports/device sanity check.
- `micro-smoke`: optional ultra-fast loop (e.g. `episodes=40`) for sanity only; not comparable.
- `smoke`: standard screening gate (`episodes=150`, `runs=3`); go/no-go only.
- `full`: final gate (short/long suites with `runs=20` each).

## Repo map (configs + runs)

- Config selection index: `configs/INDEX.md`
- Config usage details: `configs/README.md`
- Run artifact map: `docs/runs/README.md`
- Candidate cleanup list: `docs/runs/CANDIDATES_TO_ARCHIVE_20260221.md`

## Research objective and current status (2026-02-24)

- This repo is under active `vibe coding` iteration: small deltas, quick validation loops, and strict rollback/archive discipline.
- Current stable mainline is `v7p1` (`configs/v7p1.json`).
- Active V8 iteration candidate is `v8p12` (`configs/v8p12.json`) (train-first: align progress-dist with shortest-path via `w_clearance=0`; pending train+infer smoke; do not claim until fixed-pairs full20 passes).
- `v8p8` is the previous V8 candidate (dueling + globalcnn_fusion + aux admissibility; smoke NO-GO; full gate not yet run).
- `v8p5` is the previous V8 iteration (regression PASS; infer-only: tie-break short collision=1/3).
- `v8p1` is archived as NO-GO (navdist progress distance; smoke SR regressed).
- `v7p2` to `v7p3p7` are archived failed/iterative attempts on the non-mainline branch, and `v7p1` remains the stable claim baseline.
- Final objective: make RL planning (`CNN-DDQN`) outperform classical planning (`Hybrid A*-MPC`) under fair and reproducible evaluation.
- Core optimization targets: shorter paths (`avg_path_length`), shorter path time (`path_time_s`), and smoother trajectories (`avg_curvature_1_m`, lower is better).

## Quickstart (Ubuntu/bash)

All commands below assume you run from the `dqn/` folder so outputs go to `runs/` by default:

```bash
cd /home/sun/phdproject/dqn/dqn
```

Two ways to run commands:

- Recommended (reproducible/CI-friendly): keep `conda run -n ros2py310 ...`.
- Optional (interactive shell): run `conda activate ros2py310` once, then use `python ...` directly.

Self-check (fast sanity check of imports/device):

```bash
bash scripts/self_check.sh

# (equivalent explicit commands)
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
conda run -n ros2py310 python game.py --self-check
```

Optional extras (for live training window):

```bash
conda run -n ros2py310 python -m pip install -r requirements-optional.txt
```

## Breaking changes (2026-02-06)

- Forest internal cost-to-go chain is removed; environment now uses distance-based shaping/sampling.
- Forest training expert supports `hybrid_astar` and `astar_mpc` (`--forest-expert {auto,hybrid_astar,astar_mpc}`, where `auto -> hybrid_astar`).
- Forest bicycle observation changed from `11 + 2*N*N` to `10 + N*N` (occupancy map only).
- Random-pair flags were renamed from `*-cost-m` to `*-dist-m` (old names removed).
- Infer supports suite ratio thresholds:
  - `--rand-short-min-dist-ratio`, `--rand-short-max-dist-ratio`
  - `--rand-long-min-dist-ratio`, `--rand-long-max-dist-ratio`

Old checkpoints/configs with the previous forest observation/flag schema are not backward compatible.

## Version note (2026-02-22)

- As of 2026-02-22, the stable mainline remains `v7p1`.
- `v7p2/v7p2p1` attempted a Markov-observation fix (adding `prev_a_n`) but did not show stable gains and was archived as a failed branch.
- `v7p3` introduced suite-specific no-progress penalties (short/long) in two-suite training; smoke verdict is NO-GO and it is archived as failed.
- `v7p3p1` replaced suite-specific penalties with adaptive no-progress penalty by distance ratio; smoke shows SR recovery on mid/long but path/time/smoothness regressed, so it is archived as failed.
- `v7p3p2` added turn-aware top-k replacement scoring to suppress aggressive obstacle-turning; smoke reduced path/time but SR dropped sharply, so it is archived as failed.
- `v7p3p3` tuned inference-gating params for turn-aware replacement (`tp=0.3`, `min_prog=0.0`); smoke recovered long SR but short collapsed to 0 with collision/timeout, so it is archived as failed.
- `v7p3p4` patched safe fallback for admissible gating (avoid keeping inadmissible `argmax(Q)` when progress-mask is empty) and fixed `fallback_rate`; infer-only smoke (fixed v7p3p2 checkpoint) recovered SR to `0.667/0.667/1.000` with zero collisions, but still lags baseline on short/mid SR and path/time, so it is archived as failed.
- `v7p3p6` keeps `obs_map_size=128` and applies long-recovery tuning (`forest_topk_turn_penalty=0.3`, `forest_min_progress_m=0.0`, long-biased curriculum); smoke improved long SR from `0.000` to `0.333` versus v7p3p5, but short/long still fail baseline gates, so it is archived as failed.
- `v7p3p7` keeps `obs_map_size=128` and applies timeout-cut tuning (`forest_topk=12`, `forest_topk_turn_penalty=0.2`, `forest_min_progress_m=0.02`); smoke raised short/mid SR to `1.000/1.000` and reduced total CNN timeout from `5` to `2` versus v7p3p6, but long remains `0.333` with `2/3 timeout` and short/long path-time gates still fail, so it is archived as failed.
- `v7p1` remains the stable comparison baseline (forest bicycle observation `10 + N*N`), while new module versions iterate forward on separate version tracks.
- Failure archive: `docs/versions/v7p2p1/`.

## Train / infer (recommended: config profiles)

Profiles live under `configs/*.json` and are loaded via `--profile <name>`:

```bash
conda run -n ros2py310 python train.py --profile forest_a_all6_300_cuda
conda run -n ros2py310 python infer.py --profile forest_a_all6_300_cuda
```

Equivalent commands in an activated shell (`conda activate ros2py310`):

```bash
python train.py --profile forest_a_all6_300_cuda
python infer.py --profile forest_a_all6_300_cuda
```

### Latest train/infer commands (keep updated)

Last updated: 2026-02-24  
Current recommended train profile: `v7p1`

```bash
conda run -n ros2py310 python train.py --profile v7p1
conda run -n ros2py310 python infer.py --profile v7p1
```

Active V8 smoke profiles (experimental):

```bash
# v8p9: infer sweep smoke (fixed pairs3 subset from pairs20; runs=3)
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p9_infer_sweep_short_smoke
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p9_infer_sweep_long_smoke
```

```bash
# v8p8: smoke (episodes=150, runs=3) [DONE: NO-GO]
conda run -n ros2py310 python train.py --profile repro_20260224_v8p8_smoke
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p8_smoke

# v8p7: infer-only smoke (fixed v8p6 smoke checkpoint; goal-approach speed shaping; runs=3)
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p7_goal_approach_infer_smoke

# v8p7: train+infer smoke (episodes=150, runs=3) [pending]
conda run -n ros2py310 python train.py --profile v8p7
conda run -n ros2py310 python infer.py --profile v8p7

# v8p5: regression (replace-ranking ablation on v8p3 smoke failure pairs; runs=2)
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_regression

# v8p5: infer-only smoke (fixed v7p1 checkpoint; replace-ranking ablation; runs=3)
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke --forest-replace-ranking progress_clearance_q
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke --forest-replace-ranking clearance_progress_q

# v8p6: infer-only smoke (fixed v7p1 checkpoint; replace-topq; runs=3)
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke --forest-replace-topq 1
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke --forest-replace-topq 3

# v8p6: train+infer smoke (episodes=150, runs=3) [latest: NO-GO (short/mid collision=1/3)]
conda run -n ros2py310 python train.py --profile v8p6 --forest-replace-topq 3
conda run -n ros2py310 python infer.py --profile v8p6 --forest-replace-topq 3

# v8p4: regression (replay v8p3 smoke failure pairs: mid collision + long timeout; runs=2)
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p4_smoke_failures_regression

# v8p4: train+infer smoke (episodes=150, runs=3) [on hold: regression FAIL]
conda run -n ros2py310 python train.py --profile repro_20260223_v8p4_fallback_h1_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p4_fallback_h1_smoke

# v8p2 reference: infer-only A/B (fixed v7p1 checkpoint): dijkstra8_nocorner vs euclid
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke --forest-progress-dist-mode euclid
```

Latest archived candidate (for replay, smoke NO-GO):

```bash
conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke --models runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248
```

Note: training now saves process logs to `<run_dir>/train_flow.log` by default; disable via `--no-save-train-log`.

Primary claim regime (docs/paper, current mainline): `CNN-DDQN (shielded/hybrid inference)`.
- For v6p2p3/v7p1 and onward mainline reporting, use `shielded/hybrid` naming.
- Do not present these mainline results as `strict-argmax`.

### Remote-first execution (`ubuntu-zt`)

Default policy: run train/infer on `ssh ubuntu-zt` first (including smoke/full).  
Fallback to local only when remote is unavailable.

```bash
# 1) Sync local repo -> remote repo (local is source of truth)
rsync -avz --delete \
  --exclude '.git/' \
  --exclude 'runs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  /home/sun/phdproject/dqn/dqn/ \
  ubuntu-zt:/home/sun/phdproject/dqn/dqn/

# 2) Run on remote (example: smoke train / micro-smoke, episodes=40; sanity only)
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v6p2p3 --episodes 40 --out v6p2p3_smoke --device cuda --progress"

# 3) Sync remote results -> local runs/
rsync -avz \
  ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v6p2p3_smoke/ \
  /home/sun/phdproject/dqn/dqn/runs/v6p2p3_smoke/
```

Micro-smoke (optional quick loop; not the standard smoke gate):

```bash
conda run -n ros2py310 python train.py --profile v6p2p3 --episodes 40 --out v6p2p3_smoke
conda run -n ros2py310 python infer.py --profile v6p2p3 --models v6p2p3_smoke --runs 3 --out v6p2p3_smoke
```

Fixed mid (14-42m) infer commands (strict vs hybrid, runs=20, diagnostic ablation):

```bash
conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_mid_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_mid_pairs20_v1
```

v6p1 timeout-tuned fixed pairs infer (hybrid/shielded, runs=20; checkpoint pinned in profile):

```bash
# NOTE: v6p1 long/mid gating regresses short; keep v6 for short.
conda run -n ros2py310 python infer.py --profile repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1
```

### Train live view (pygame, RL stage only)

Default is off. Enable it explicitly during training:

```bash
conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --live-view --live-view-fps 0 --live-view-window-size 900 --live-view-trail-len 300 --live-view-skip-steps 1
```

- Viewer is attached only to RL training episodes (`env+algo`); demo collect/pretrain are not shown.
- Closing the pygame window does not stop training; it auto-falls back to no-view mode.
- If `pygame` is not installed, training continues and prints an install hint.
- Fixed-size vehicle collision box (oriented by heading from `pose_m`) is ON by default; use `--no-live-view-collision-box` to hide it.

### Interactive goal-click game (pygame)

Left-click a goal on the map, pick a planner, and let `mpc` track the planned path.

```bash
conda run -n ros2py310 python game.py --profile repro_20260212_interactive_game_forest_a_v1
```

Planner hotkeys: `1`=hybrid A*, `2`=RRT*, `3`=grid A*, `4`=cnn-ddqn (requires `--rl-checkpoint <path>`).  
Other: `R` reset, `SPACE` pause, `P` replan.

## `runs/` layout and report artifacts

Output routing:
- If `--out <name>` is a bare name, outputs go to `runs/<name>/`.
- If `--out <path>` is a path, outputs go to `<path>/` as-is.

Typical structure (train + nested infer):

```text
runs/<out>/
  latest.txt
  train_YYYYMMDD_HHMMSS/
    configs/run.json
    train_flow.log
    models/
    infer/
      latest.txt
      YYYYMMDD_HHMMSS/
        configs/run.json
        table2_kpis_mean_raw.csv
        table2_kpis_raw.csv
```

Notes:
- For scripts/machines, prefer `*_raw.csv` files. The non-raw `table2_kpis_mean.csv` uses human-friendly column names.
- Final gates/tables should read `table2_kpis_mean_raw.csv`; failure distribution/diagnostics should read `table2_kpis_raw.csv`.

### KPI dictionary (`table2_kpis_*` columns, minimal)

- `success_rate`: success ratio in `[0,1]` (higher is better).
- `avg_path_length`: average planned path length in meters (lower is better).
- `path_time_s`: trajectory execution time in seconds (lower is better).
- `avg_curvature_1_m`: average curvature in `1/m` (lower is smoother).
- `planning_time_s`: planner time in seconds (lower is better).
- `tracking_time_s`: tracking/controller time in seconds (lower is better).
- `inference_time_s`: policy inference time in seconds (RL only; lower is better).
- `argmax_inadmissible_rate`: fraction of steps where `argmax(Q)` is inadmissible (diagnostic).
- `fallback_rate`: fraction of steps where inference-time fallback/override triggered (diagnostic; should be `0` in `strict-argmax` by definition).
- `failure_reason`: failure type label (only in `table2_kpis_raw.csv`).

## 版本总索引（v1 → v8p15）

> 说明：本索引用于统一 `docs/versions/` 的重编号口径；历史目录 `v3p1`~`v3p11` 保留原记录，未纳入本轮重编号；早期误混入版本链已于 2026-02-09 清理。当前稳定主线为 `v7p1`；`v8p11` smoke 显示 short 已可在 SR=1.0 下压过 baseline，但 long 仍落后；`v8p12/v8p13` 为 NO-GO；`v8p14/v8p15` infer-only sweep 虽可维持 long 的 SR=1.0，但仍无法压过 baseline（详见 `docs/versions/README.md`）。

| 版本 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|
| `v1` | `docs/versions/v1/` | `configs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke.json` | `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017` | `0.0 / 0.0` | `1.0 / 1.0` | 未通过 |
| `v2` | `docs/versions/v2/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246` | `0.0 / 0.0` | `1.0 / 1.0` | 未通过 |
| `v3` | `docs/versions/v3/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403` | `0.5 / 0.1` | `0.9 / 1.0` | 未通过 |

### 增量版本（v3p1 → v8p15）

| 版本 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|
| `v3p12` | `docs/versions/v3p12/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622` | `0.0 / 0.0` | `0.95 / 1.0` | 未通过 |
| `v4p1` | `docs/versions/v4p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10/20260210_135524` | `0.1 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p2` | `docs/versions/v4p2/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730` | `0.0 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p3` | `docs/versions/v4p3/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934` | `0.2 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p3p1` | `docs/versions/v4p3p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044` | `0.0 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v5` | `docs/versions/v5/` | `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1.json` | `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351` | `0.75 / 0.85` | `0.95 / 0.90` | 未通过 |
| `v6` | `docs/versions/v6/` | `configs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602` | `0.90 / 0.70` | `0.95 / 0.90` | 未通过 |
| `v6p1` | `docs/versions/v6p1/` | `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414` | `0.70 / 0.95` | `0.95 / 0.90` | 未通过 |
| `v6p2` | `docs/versions/v6p2/` | `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414` | `0.70 / 0.95` | `0.95 / 0.90` | 未通过 |
| `v6p2p2` | `docs/versions/v6p2p2/` | `configs/v6p2p2.json` | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433` | `0.75 / 0.55` | `0.95 / 1.00` | 未通过（待 full） |
| `v6p2p3` | `docs/versions/v6p2p3/` | `configs/v6p2p3.json` | `runs/v6p2p3/train_20260219_142104/infer/20260219_145315` | `0.80 / 1.00` | `1.00 / 1.00` | 已运行（runs=5，待 full20） |
| `v7p1` | `docs/versions/v7p1/` | `configs/v7p1.json` | `runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927` | `1.00 / 1.00` | `1.00 / 1.00` | 稳定主线（runs=5，待 full20） |
| `v7p2` | `docs/versions/v7p2/` | `configs/v7p2.json` | `runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137` | `1.00 / 1.00` | `1.00 / 1.00` | 已运行（micro-smoke：episodes=40, runs=3） |
| `v7p2p1` | `docs/versions/v7p2p1/` | `configs/repro_20260220_v7p2p1_rollback_v7p1.json` | `runs/v7p2_es150/train_20260220_222056/infer/20260220_223016` | `0.85 / 0.65` | `0.95 / 1.00` | 失败归档，已回退到 `v7p1` |
| `v7p2p2` | `docs/versions/v7p2p2/` | `configs/repro_20260221_v7p2p2_globalcnn_smoke.json` | `runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（smoke 不达门，主线保持 `v7p1`） |
| `v7p2p3` | `docs/versions/v7p2p3/` | `configs/repro_20260221_v7p2p3_globalcnn_fusion_smoke.json` | `runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334/infer/20260221_180256` | `0.333 / 0.667` | `1.00 / 1.00` | 失败归档（smoke 不达门，主线保持 `v7p1`） |
| `v7p2p4` | `docs/versions/v7p2p4/` | `configs/repro_20260221_v7p2p4_globalcnn_spatialprior_smoke.json` | `runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926` | `0.667 / 1.000` | `1.00 / 1.00` | 失败归档（smoke 不达门，保持当前代码并继续前向迭代） |
| `v7p2p5` | `docs/versions/v7p2p5/` | `configs/repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke.json` | `runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626` | `0.333 / 0.667` | `1.00 / 1.00` | 失败归档（smoke 退化，不回退代码并继续前向迭代） |
| `v7p2p6` | `docs/versions/v7p2p6/` | `configs/repro_20260221_v7p2p6_foundationfix_smoke.json` | `runs/v7p2p6_foundationfix_smoke/train_20260221_211603/infer/20260221_213248` | `1.000 / 0.000` | `1.00 / 1.00` | 失败归档（short 改善但 long 崩塌，继续前向迭代） |
| `v7p2p7` | `docs/versions/v7p2p7/` | `configs/repro_20260221_v7p2p7_gradclip_recover_smoke.json` | `runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452/infer/20260221_221008` | `0.333 / 0.333` | `1.00 / 1.00` | 失败归档（long 有恢复但 short 退化，继续前向迭代） |
| `v7p2p8` | `docs/versions/v7p2p8/` | `configs/repro_20260221_v7p2p8_bold_dynamic_expert_smoke.json` | `runs/v7p2p8_bold_dynamic_expert_smoke/train_20260221_225358/infer/20260221_230426` | `0.000 / 1.000` | `1.00 / 1.00` | 失败归档（long 恢复到 1.0，但 short 崩塌到 0.0，继续前向迭代） |
| `v7p2p9` | `docs/versions/v7p2p9/` | `configs/repro_20260221_v7p2p9_ablate_expert_smoke.json` | `runs/v7p2p9_ablate_expert_smoke/train_20260221_231402/infer/20260221_232825` | `0.667 / 0.000` | `1.00 / 1.00` | 失败归档（short 回升但 long 崩塌，继续前向迭代） |
| `v7p2p10` | `docs/versions/v7p2p10/` | `configs/repro_20260221_v7p2p10_penalty035_smoke.json` | `runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（long 回升但 short 路径与平滑性退化，继续前向迭代） |
| `v7p3` | `docs/versions/v7p3/` | `configs/repro_20260221_v7p3_suite_penalty_smoke.json` | `runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（short/mid 局部改善但 long path/time 退化，未过 smoke 门） |
| `v7p3p1` | `docs/versions/v7p3p1/` | `configs/repro_20260222_v7p3p1_adaptive_penalty_smoke.json` | `runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552` | `0.667 / 1.000` | `1.00 / 1.00` | 失败归档（mid/long SR 提升至 1.0，但 path/time/smoothness 全面退化） |
| `v7p3p2` | `docs/versions/v7p3p2/` | `configs/repro_20260222_v7p3p2_turnaware_smoke.json` | `runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842` | `0.333 / 0.333` | `1.00 / 1.00` | 失败归档（路径/时间有所回落，但三套件 SR 显著下降，未过 smoke 门） |
| `v7p3p3` | `docs/versions/v7p3p3/` | `configs/repro_20260222_v7p3p3_infergate_smoke.json` | `runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657` | `0.000 / 0.667` | `1.00 / 1.00` | 失败归档（long SR 回升，但 short=0 且出现碰撞/超时，未过 smoke 门） |
| `v7p3p4` | `docs/versions/v7p3p4/` | `configs/repro_20260222_v7p3p4_safe_fallback_infer_smoke.json` | `runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513` | `0.667 / 1.000` | `1.00 / 1.00` | 失败归档（safe fallback 补丁修复碰撞回潮；但 short/mid SR 仍落后 baseline，且 path/time 更差；infer-only smoke） |
| `v7p3p6` | `docs/versions/v7p3p6/` | `configs/repro_20260222_v7p3p6_obsmap128_tune_smoke.json` | `runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007/infer/20260222_223831` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（long 从 0.000 回升到 0.333，但 short/long 仍未过门） |
| `v7p3p7` | `docs/versions/v7p3p7/` | `configs/repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke.json` | `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329` | `1.000 / 0.333` | `1.00 / 1.00` | 失败归档（short/mid SR 升至 1.0 且 CNN 总 timeout 从 5 降到 2；但 long 仍 2/3 timeout，short/long path-time 仍落后 baseline） |
| `v8p1` | `docs/versions/v8p1/` | `configs/v8p1.json` | `runs/v8p1_navdist_smoke/train_20260223_021339/infer/20260223_023932` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（navdist progress distance；smoke SR 退化） |
| `v8p2` | `docs/versions/v8p2/` | `configs/v8p2.json` | `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027` | `0.667 / 1.000` | `1.00 / 1.00` | smoke 已跑（mid/long=1.0；short=2/3 collision；暂不 full） |
| `v8p3` | `docs/versions/v8p3/` | `configs/v8p3.json` | `runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153` | `1.000 / 0.667` | `1.00 / 1.00` | 失败归档（smoke：mid collision=1/3；long timeout=1/3） |
| `v8p4` | `docs/versions/v8p4/` | `configs/v8p4.json` | `runs/v8p4_smoke_failures_regression/20260223_142739` | `N/A / N/A` | `N/A / N/A` | 失败归档（回归 FAIL：collision+timeout；暂不 smoke） |
| `v8p5` | `docs/versions/v8p5/` | `configs/v8p5.json` | `runs/v8p5_replace_ranking_infer_smoke/20260223_172217` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only：`q` PASS；tie-break short `collision=1/3`（NO-GO）；train+infer smoke 未跑 |
| `v8p6` | `docs/versions/v8p6/` | `configs/v8p6.json` | `runs/v8p6_replace_topq_infer_smoke/20260223_185628` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only：topq=1/2/3 均 PASS（推荐 topq=3）；train+infer smoke NO-GO（short/mid collision=1/3） |
| `v8p7` | `docs/versions/v8p7/` | `configs/v8p7.json` | `runs/v8p7_goal_approach_infer_smoke/20260223_230524` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only：goal-approach speed shaping PASS（SR=1.0）；train+infer smoke 待跑 |
| `v8p8` | `docs/versions/v8p8/` | `configs/v8p8.json` | `runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556` | `0.667 / 1.000` | `1.00 / 1.00` | smoke 已跑（NO-GO；short SR 低于 baseline；mid/long path/time 劣于 baseline） |
| `v8p9` | `docs/versions/v8p9/` | `configs/v8p9.json` | `runs/v8p9_infer_sweep_short_pairs3_smoke/20260224_114743` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only sweep smoke 已跑（pairs3：SR=1.0 可达；但 path/time 仍落后 baseline；full 暂不建议） |
| `v8p10` | `docs/versions/v8p10/` | `configs/v8p10.json` | `runs/v8p10_sweep_long_w2p0/20260224_135035` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only sweep smoke 已跑（pairs3：SR=1.0 可达；w_clearance sweep 有回落但仍落后 baseline；full 暂不建议） |
| `v8p11` | `docs/versions/v8p11/` | `configs/v8p11.json` | `runs/v8p11_infer_smoke_long_pairs3/20260224_152917` | `1.000 / 1.000` | `1.00 / 1.00` | smoke 已跑：short 打败 baseline；long 仍落后（另有 long w_clearance sweep 归档）；full 暂不建议 |
| `v8p12` | `docs/versions/v8p12/` | `configs/v8p12.json` | `runs/v8p12_infer_smoke_long_pairs3/20260224_163613` | `1.000 / 1.000` | `1.00 / 1.00` | smoke 已跑：short 明显回退；long detour 仅小幅回落但仍落后 baseline（NO-GO；full 暂不建议） |
| `v8p13` | `docs/versions/v8p13/` | `configs/v8p13.json` | `runs/v8p13_infer_smoke_long_pairs3/20260224_172037` | `1.000 / 1.000` | `1.00 / 1.00` | smoke 已跑：long path/time 显著退化（NO-GO；full 暂不建议） |
| `v8p14` | `docs/versions/v8p14/` | `configs/v8p14.json` | `runs/v8p14_sweep_long_pairs3_w1p5_mpneg005/20260224_174542` | `N/A / 1.000` | `N/A / 1.00` | infer-only sweep 已跑：SR 恢复但 long 仍落后 baseline（NO-GO；full 暂不建议） |
| `v8p15` | `docs/versions/v8p15/` | `configs/v8p15.json` | `runs/v8p15_sweep_long_pairs3_sigma0p3/20260224_180520` | `N/A / 1.000` | `N/A / 1.00` | infer-only sweep 已跑：sigma=0.3 最优但 long 仍落后 baseline（NO-GO；full 暂不建议） |

- baseline-only（`--skip-rl`）输出不计入上表；请单独查看 `runs/outputs_forest_baselines/*`、`runs/repro_20260207_*` 等目录。
- 详细四件套请见 `docs/versions/README.md` 与各版本目录。

## Rigour and anti-cheating rules (mandatory)

- RL vs baseline comparison must use the same environment/suite and fixed start-goal pairs (no sample drift).
- Claiming gains requires matched evaluation budget and protocol; final claims must use short/long suites with `runs=20` each.
- Smoke results (`episodes=150`, `runs=3`) are screening-only and cannot be used as final claims.
- Inference policy naming must match implementation (`strict-argmax` vs `shielded/masked/hybrid`), with no hidden intervention.
- No cherry-picking: failed versions/runs must be archived, and missing outputs must be reported as `N/A` with reasons.
- Every claim must be traceable to artifacts: command line, `run_dir`, `run.json`, and `table2_kpis_mean_raw.csv`.

## Recommended iteration workflow (version-first)

For each new version candidate, use this default pipeline:

1. Pre-version snapshot (mandatory):
- Ensure clean workspace: `git status`.
- Snapshot and push before changes:
```bash
git add -A
git commit -m "snapshot: pre-<version>"
git push origin <branch>
```

2. Implement one small version delta only (single-purpose change).

3. Smoke gate (fixed quick loop):
```bash
conda run -n ros2py310 python train.py --profile <candidate> --episodes 150 --out <version>_smoke150 --device cuda
conda run -n ros2py310 python infer.py --profile <candidate> --models <version>_smoke150 --runs 3 --out <version>_smoke150
```

4. Go/No-Go rule:
- If smoke does not show clear gain, stop escalation and mark as failed version.
- Keep the latest code as current baseline (no rollback), and continue with a single-purpose forward iteration.

5. Archive immediately:
- Create `docs/versions/<version>/` four-doc bundle and log commands, run paths, KPIs, and failure reasons.
- Prepare next iteration as `<version+1>` (example: `v7p3p5`).

## Final acceptance gate (short/long suites, runs=20)

Final claims must be reported on both short and long suites with `runs=20` per suite.

Use `table2_kpis_mean_raw.csv` and compare `CNN-DDQN` against `Hybrid A*-MPC` with all conditions below:

- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

Also report smoothness in the same table via `avg_curvature_1_m` (lower is smoother). This is a mandatory reporting metric and optimization target for version selection.

If any suite fails any condition above, the final gate is considered failed.

Inference regime naming note: this repo distinguishes `strict-argmax` (legacy label: strict no-fallback) vs `shielded/masked/hybrid`. `strict-argmax` means pure `argmax(Q)` inference (no masking/top-k/stop-override/replacement/heuristic takeover/planner takeover); masks may be computed for logging only. If any inference-time intervention is enabled, label it `shielded/masked/hybrid` (not `strict-argmax`). Current mainline claim/reporting regime is `shielded/hybrid`.

### Strict-argmax vs hybrid re-eval (fixed pairs, diagnostic only)

To compare checkpoints fairly without random-pair drift, evaluate on **fixed random pairs** (short/long suites) and report two regimes. This section is for ablation/diagnostics; it is not the primary claim template for current mainline versions.

- `strict-argmax`: pass `--forest-no-fallback` (pure `argmax(Q)`).
- `hybrid/shielded`: pass `--no-forest-no-fallback` (allows stop-override + replacement only; no heuristic fallback).

Fixed pairs (forest_a, short/long suites, 20 each):

- `configs/repro_20260210_forest_a_pairs_short20_v1.json`
- `configs/repro_20260210_forest_a_pairs_long20_v1.json`

Template (reuse a profile to keep env/action-space settings consistent with the checkpoint):

```bash
PROFILE=repro_20260211_forest_a_cnn_ddqn_v5_smoke
MODELS_DIR="runs/<exp>/<train_timestamp>/models"

# strict-argmax (short)
conda run -n ros2py310 python infer.py --profile "$PROFILE" --baselines \\
  --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 \\
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \\
  --models "$MODELS_DIR" --out repro_reval_strict_short_pairs20 \\
  --forest-no-fallback

# hybrid/shielded (short)
conda run -n ros2py310 python infer.py --profile "$PROFILE" --baselines \\
  --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 \\
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \\
  --models "$MODELS_DIR" --out repro_reval_hybrid_short_pairs20 \\
  --no-forest-no-fallback
```

For the long suite, replace `forest_a::short` + `pairs_short20` with `forest_a::long` + `pairs_long20`.

## Demonstrations (DQfD)

By default, training uses `--demo-mode dqfd` (strict DQfD-style):

- Prioritized experience replay (PER) + importance sampling (IS) weights
- 1-step TD loss + n-step TD loss + large-margin expert loss + L2 regularization
- No behavior cloning CE loss (to match the DQfD definition)

To reproduce the previous demo stabilizer behavior, use `--demo-mode legacy`.

For A*+MPC expert training with curve optimization (shortcut + resample + minimum-turn-radius + dual-circle collision checks),
use `--forest-expert astar_mpc` together with `--forest-astar-opt-*` and `--forest_mpc_*` flags, or load profile:

```bash
conda run -n ros2py310 python train.py --profile repro_20260208_forest_a_cnn_ddqn_dqfd_astar_mpc_curveopt_300
```

Paper copies + BibTeX are archived under `paper/dqfd_refs/`.

## Baseline eval (no RL checkpoints required)

`--baselines all` now includes six baselines (fixed order):

1. `astar`
2. `hybrid_astar`
3. `rrt_star`
4. `astar_mpc` (`A*-MPC`)
5. `hybrid_astar_mpc` (`Hybrid A*-MPC`)
6. `rrt_mpc` (`RRT-MPC`)

Run all six on CPU:

```bash
conda run -n ros2py310 python infer.py --envs forest_a --out outputs_forest_baselines --baselines all --skip-rl --runs 5 --device cpu
```

Run only MPC-combo baselines:

```bash
conda run -n ros2py310 python infer.py --envs forest_a --out outputs_forest_mpc_baselines --baselines astar_mpc hybrid_astar_mpc rrt_mpc --skip-rl --runs 5 --device cpu
```

### Fixed random pairs (fair baseline comparison)

Use the frozen random-pair profile to compare planners on the exact same `(start, goal)` samples:

```bash
conda run -n ros2py310 python infer.py --profile repro_20260206_6baselines_fair_forest_a_fixedpairs --skip-rl
```

This profile reads pairs from `configs/repro_20260206_6baselines_fair_forest_a_pairs.json`.

Suite-split fixed pairs (short/long, 20 each):

- `configs/repro_20260210_forest_a_pairs_short20_v1.json`
- `configs/repro_20260210_forest_a_pairs_long20_v1.json`

Legacy `forest_baseline_mpc_*` profile keys are ignored during infer config loading (deprecated).

## Success definition

Forest bicycle success is:

- `reached_pose`: within goal tolerance, and (optionally) within a heading-to-goal tolerance
- `reached_stop`: stopped and nearly straight wheels (near-zero `|v|` and `|delta|`)
- `reached` / "success" == `reached_pose AND reached_stop`

Implemented in `forest_vehicle_dqn/env.py` (`AMRBicycleEnv._step_with_controls`, `_goal_pose_reached`, `_goal_stop_reached`).

More runnable examples + flag reference: [`runtxt.md`](runtxt.md).
