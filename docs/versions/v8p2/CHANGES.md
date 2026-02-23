# v8p2 变更清单（相对 v8p1）

## 1) 代码改动

### `forest_vehicle_dqn/env.py`
- 新增 `dijkstra8_nocorner_goal_dist_m(...)`（8 邻接 + 禁止穿角的 Dijkstra 距离场；支持 costmap 权重）。
- `AMRBicycleEnv(progress_dist_mode=...)` 扩展 `forest_progress_dist_mode`：
  - 新增 `dijkstra8_nocorner`
- 新增 costmap 参数：
  - `progress_cost_w_clearance`（clearance 惩罚权重）
  - `progress_cost_sigma_m`（惩罚衰减长度，m）
- progress 距离采样改为 finite-safe（避免 `inf/NaN` 传导到 reward/gating/fallback）。

### `forest_vehicle_dqn/cli/train.py` / `forest_vehicle_dqn/cli/infer.py`
- `--forest-progress-dist-mode` choices 增加 `dijkstra8_nocorner`。
- 新增 CLI 参数并透传到环境：
  - `--forest-progress-cost-w-clearance`
  - `--forest-progress-cost-sigma-m`

## 2) 测试改动

### `tests/test_forest_progress_distance.py`
- 新增 Dijkstra（对角步长、禁穿角、加权代价累积）与 finite-safe 插值的单测覆盖。

## 3) 配置与文档

- 新增：
  - `configs/v8p2.json`
  - `configs/repro_20260223_v8p2_costmap_smoke.json`
  - `configs/repro_20260223_v8p2_costmap_infer_smoke.json`
  - `docs/plans/2026-02-23-v8p2-costmap-dijkstra-design.md`
  - `docs/plans/2026-02-23-v8p2-costmap-dijkstra-implementation-plan.md`
  - `docs/versions/v8p2/`（四件套）
- 更新：
  - `configs/INDEX.md`（V8 迭代入口切换到 `v8p2`）

