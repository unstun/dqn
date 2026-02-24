# v8p12 变更清单（相对 v8p11）

## 1) 主要策略（long detour 优先：口径对齐）

- 将 `forest_progress_cost_w_clearance`（Dijkstra cost 的 clearance 惩罚权重）在 train+infer 中统一设为 `0.0`：
  - 目标：让 `progress_dist_mode=dijkstra8_nocorner` 的 cost-to-go 更接近最短路距离，减少 long 绕路。
- 将 `forest_demo_w_clearance`（demo/专家采样时的 clearance 代价权重）设为 `0.0`：
  - 目标：降低专家路径偏好与训练奖励进度口径的冲突，提升 long 上“短路径”学习信号。

反作弊约束：
- 不改 `goal_tolerance_m` / stop 阈值；baseline 必须同跑。

## 2) 配置与文档

- 新增 `configs/v8p12.json`（版本入口）
- 新增 `configs/repro_20260224_v8p12_train_smoke.json`（训练 smoke，可复现）
- 新增 `configs/repro_20260224_v8p12_infer_smoke_{short,long}.json`（推理 smoke，可复现）
- 新增 `docs/versions/v8p12/` 四件套（本文件为变更明细）

## 3) 受影响文件清单

- `configs/v8p12.json`
- `configs/repro_20260224_v8p12_train_smoke.json`
- `configs/repro_20260224_v8p12_infer_smoke_short.json`
- `configs/repro_20260224_v8p12_infer_smoke_long.json`
- `docs/versions/v8p12/README.md`
- `docs/versions/v8p12/CHANGES.md`
- `docs/versions/v8p12/RESULTS.md`
- `docs/versions/v8p12/runs/README.md`

