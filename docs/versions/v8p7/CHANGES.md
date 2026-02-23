# v8p7 变更清单（相对 v8p6）

## 1) 代码改动

- `forest_vehicle_dqn/cli/infer.py`
  - 新增 `--forest-goal-approach-override`（接近目标速度整形开关）
  - 新增 `--forest-goal-approach-dist-m`（触发距离阈值；<=0 采用 `2.5*goal_tolerance_m`）
  - 新增 `--forest-goal-approach-speed-factor`（速度包络系数；<=0 采用 `0.8`）
  - `rollout_agent(...)`（用当前策略在环境里采样轨迹/回合）在接近 goal 且未 stop 时，对动作做推理侧速度整形：保持 `delta_dot` 不变，仅在同 `delta_dot` 动作子集里挑 admissible 的 `accel` 以降低 `|v_next|`。
  - `goal_approach_override_steps`（接近目标整形步数计数）写入 rollout debug（用于诊断，不进入 KPI 表）。

## 2) 配置改动

- 新增 `configs/v8p7.json`（V8 迭代入口，默认启用 `forest_goal_approach_override`）
- 新增 `configs/repro_20260223_v8p7_goal_approach_infer_smoke.json`（固定 v8p6 checkpoint 的 infer-only smoke 可复现快照）

## 3) 文档改动

- 新增 `docs/versions/v8p7/` 四件套与代表 run 映射
- 更新索引与 README：把 active V8 入口从 `v8p6` 更新为 `v8p7`

## 4) 受影响文件清单

- `forest_vehicle_dqn/cli/infer.py`
- `configs/v8p7.json`
- `configs/repro_20260223_v8p7_goal_approach_infer_smoke.json`
- `docs/versions/v8p7/README.md`
- `docs/versions/v8p7/CHANGES.md`
- `docs/versions/v8p7/RESULTS.md`
- `docs/versions/v8p7/runs/README.md`
- `configs/INDEX.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`

