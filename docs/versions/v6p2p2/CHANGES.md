# v6p2p2 - 变更

## 版本意图
- 围绕 `forest_reward_k_t`（时间惩罚系数）与 `forest_reward_k_delta`（方向变化平滑惩罚系数）做系统化参数搜索，形成可复现实验结论。

## 相对 v6p2 的代码/配置变更
- 新增训练参数开关（不改变默认行为）：
  - `forest_vehicle_dqn/cli/train.py`
    - 新增 `--forest-reward-k-t`（默认 `0.1`）
    - 新增 `--forest-reward-k-delta`（默认 `1.5`）
    - 在 `AMRBicycleEnv(...)` 注入 `reward_k_t` 与 `reward_k_delta`
- 新增 sweep 自动化脚本：
  - `scripts/sweep_v6p2p2_reward_grid.py`
- 新增本轮实验 profile：
  - `configs/repro_20260219_v6p2p2_reward_kt_kdelta_sweep_smoke.json`
  - `configs/repro_20260219_v6p2p2_reward_kt_kdelta_sweep_full.json`
- 新增建议参数 profile：
  - `configs/v6p2p2.json`

## 文档与归档变更
- 新增版本四件套：
  - `docs/versions/v6p2p2/README.md`
  - `docs/versions/v6p2p2/CHANGES.md`
  - `docs/versions/v6p2p2/RESULTS.md`
  - `docs/versions/v6p2p2/runs/README.md`
- 索引同步：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`

## 未改动项
- 未新增依赖；未改动推理决策口径命名（仍按 `shielded/hybrid` 留档）。
