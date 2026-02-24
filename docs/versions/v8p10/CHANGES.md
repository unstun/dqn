# v8p10 变更清单（相对 v8p9）

## 1) 主要策略（推理优先：progress-dist clearance 消融）

- 固定 `forest_progress_dist_mode=dijkstra8_nocorner`（障碍感知最短路 cost-to-go），并对 `forest_progress_cost_w_clearance`（靠近障碍的代价权重）做推理侧消融 sweep，目标是在 SR≈1.0 前提下压 `avg_path_length/path_time_s`，追平/反超 baseline（Hybrid A*-MPC）。
- 同时引入最小代码改动：新增 progress-first 的 replacement ranking（减少因 clearance 优先导致的绕路），并将 v8p10 默认推理侧 `forest_replace_ranking` 设为 `progress_q`。

## 2) 配置与文档

- 新增 `configs/v8p10.json`（版本入口：默认 `dijkstra8_nocorner` + `progress_cost_w_clearance=0.0`）
- 新增 `configs/repro_20260224_v8p10_infer_sweep_{short,long}_smoke.json`（推理侧 sweep smoke，可复现）
- 新增 `docs/versions/v8p10/` 四件套（本文件为变更明细）

## 3) 代码改动（推理/训练一致）

- `forest_replace_ranking` 新增两种模式：
  - `progress_q`：先最小化 next-step `progress_dist`，再最大化 Q
  - `progress_q_clearance`：先最小化 next-step `progress_dist`，再最大化 Q，最后最大化 clearance（OD）
- 目标：在可采纳候选均安全时，减少因 clearance 优先带来的绕路。

## 4) 受影响文件清单

- `configs/v8p10.json`
- `configs/repro_20260224_v8p10_infer_sweep_short_smoke.json`
- `configs/repro_20260224_v8p10_infer_sweep_long_smoke.json`
- `configs/INDEX.md`
- `forest_vehicle_dqn/cli/infer.py`
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_v8p10_replace_ranking_progress_q.py`
- `docs/versions/v8p10/README.md`
- `docs/versions/v8p10/CHANGES.md`
- `docs/versions/v8p10/RESULTS.md`
- `docs/versions/v8p10/runs/README.md`
