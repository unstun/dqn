# v8p7 runs 记录（可追溯路径）

## 1) 代表 run（infer-only smoke）

- run_dir：`runs/v8p7_goal_approach_infer_smoke/20260223_230524`
  - run_json：`runs/v8p7_goal_approach_infer_smoke/20260223_230524/configs/run.json`
  - kpi_mean_raw：`runs/v8p7_goal_approach_infer_smoke/20260223_230524/table2_kpis_mean_raw.csv`
  - kpi_raw：`runs/v8p7_goal_approach_infer_smoke/20260223_230524/table2_kpis_raw.csv`

说明：
- 本次 smoke 使用固定模型：`runs/v8p6_replace_topq_smoke/train_20260223_191450/models`
- 关键开关：`forest_replace_topq=3` + `forest_goal_approach_override=true`（`dist_m=2.5`，`speed_factor=0.8`）

## 2) 其他 run

- `N/A`

