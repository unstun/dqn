# v8p1 改动清单（相对 v7p1）

> 目标：在不改变 random start/goal 采样与套件划分口径的前提下，引入 obstacle-aware 的 progress 距离定义，用于 reward progress 与推理期 admissible gating，争取在 `SR≈1.0` 前提下压 `avg_path_length/path_time_s`。

## 1) 代码改动

- `forest_vehicle_dqn/env.py`
  - 新增 `grid4_goal_dist_m(...)`（4 邻接 BFS 最短路距离场，单位 m）
  - `AMRBicycleEnv` 新增 `progress_dist_mode`（progress 距离口径开关，`euclid|grid4`）
  - reward progress / admissible gating / fallback 动作选择切换到 progress 距离（`grid4` 时 obstacle-aware）
  - 保持采样/课程/套件筛选仍使用 `_goal_dist_m`（欧氏距离场）不变
- `forest_vehicle_dqn/cli/train.py`
  - 新增 `--forest-progress-dist-mode`（透传到 `AMRBicycleEnv(progress_dist_mode=...)`）
- `forest_vehicle_dqn/cli/infer.py`
  - 新增 `--forest-progress-dist-mode`（透传到 `AMRBicycleEnv(progress_dist_mode=...)`）

## 2) 测试
- `tests/test_forest_progress_distance.py`
  - 覆盖 `grid4_goal_dist_m(...)`：绕行（带缺口墙）与不可达区域返回 `inf`

## 3) 配置
- `configs/v8p1.json`（新增）
  - v8p1 smoke 配置入口（episodes=150, runs=3）
  - 关键开关：`forest_progress_dist_mode="grid4"`
- `configs/repro_20260222_v8p1_navdist_smoke.json`（新增）
  - v8p1 可复现 smoke（train+infer）
- `configs/repro_20260222_v8p1_navdist_infer_smoke.json`（新增）
  - v8 推理期对照（固定 v7p1 checkpoint，grid4 vs euclid）

## 4) 文档
- 设计/计划：
  - `docs/plans/2026-02-22-v8-navdist-design.md`
  - `docs/plans/2026-02-22-v8-navdist-implementation-plan.md`
- 版本四件套（新增）：
  - `docs/versions/v8p1/README.md`
  - `docs/versions/v8p1/CHANGES.md`
  - `docs/versions/v8p1/RESULTS.md`
  - `docs/versions/v8p1/runs/README.md`

