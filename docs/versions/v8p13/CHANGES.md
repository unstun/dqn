# v8p13 变更清单（相对 v8p12）

## 1) 代码改动

- `forest_vehicle_dqn/cli/train.py`：
  - 新增 `--forest-reward-k-p`（`k_p`：进度奖励系数）、`--forest-reward-k-o`（`k_o`：近障惩罚系数）、`--forest-reward-k-v`（`k_v`：近障速度耦合惩罚系数）。
  - 训练期构造 `AMRBicycleEnv(...)` 时透传 `reward_k_p/reward_k_o/reward_k_v`，便于 reward 塑形消融与 sweep。

## 2) 配置改动

- 新增版本入口：
  - `configs/v8p13.json`
- 新增可复现 smoke 配置（训练+推理）：
  - `configs/repro_20260224_v8p13_train_smoke.json`
  - `configs/repro_20260224_v8p13_infer_smoke_short.json`
  - `configs/repro_20260224_v8p13_infer_smoke_long.json`

## 3) 文档归档

- 新增版本四件套目录：
  - `docs/versions/v8p13/README.md`
  - `docs/versions/v8p13/CHANGES.md`
  - `docs/versions/v8p13/RESULTS.md`
  - `docs/versions/v8p13/runs/README.md`

