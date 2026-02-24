# v8p8 变更清单（相对 v8p7）

## 1) 算法/工程改动

- DQN 家族新增 `dueling`（Dueling DQN：V/A 分支合成 Q，默认关闭，仅 v8p8 profile 打开）
- 训练侧新增 CLI：`--dueling`、`--dueling-hidden-dim`
- 训练侧启用更强全局表征：`cnn_backbone=globalcnn_fusion` + `cnn_global_spatial_prior=true`
- 训练侧启用 `aux_admissibility_lambda`（可行性辅助监督）
- 推理侧沿用 `v8p7` 的 `forest_goal_approach_override`（接近目标速度整形），并将 `forest_goal_approach_speed_factor` 初始设为 `0.9`（后续按 full gate 结果决定是否回调）

## 2) 配置与文档

- 新增 `configs/v8p8.json`（版本入口）
- 新增 `configs/repro_20260224_v8p8_smoke.json`（smoke 可复现快照）
- 新增 `configs/pairs_v8p8_smoke_short3_20260224_110556.json` / `configs/pairs_v8p8_smoke_long3_20260224_110556.json`（推理侧消融用固定 pairs3，避免 sample drift）
- 新增 `docs/versions/v8p8/` 四件套（本文件为变更明细）
- 更新索引与 README，把 `v8p8` 登记为“待 smoke / 待 full gate”的候选版本

## 3) 受影响文件清单

- `forest_vehicle_dqn/cli/train.py`
- `configs/v8p8.json`
- `configs/repro_20260224_v8p8_smoke.json`
- `docs/versions/v8p8/README.md`
- `docs/versions/v8p8/CHANGES.md`
- `docs/versions/v8p8/RESULTS.md`
- `docs/versions/v8p8/runs/README.md`
- `configs/INDEX.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
