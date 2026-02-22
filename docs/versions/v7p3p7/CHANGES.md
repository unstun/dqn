# v7p3p7 改动清单（相对 v7p3p6）

## 变更目标
- 在不改算法定义与网络模块的前提下，针对 `obs_map_size=128` 下 long 超时偏高问题做参数纠偏（smoke 优先）。

## 配置改动明细

### 1) 主版本配置
- `configs/v7p3p7.json`（新增）
  - 基于 `v7p3p6` 重建并改名为 `v7p3p7`
  - `train.out`: `v7p3p6_obsmap128_tune_smoke` -> `v7p3p7_obsmap128_timeoutcut_smoke`
  - `infer.models/out`: `v7p3p6_obsmap128_tune_smoke` -> `v7p3p7_obsmap128_timeoutcut_smoke`
  - `train.forest_topk`: `10` -> `12`
  - `infer.forest_topk`: `10` -> `12`
  - `train.forest_topk_turn_penalty`: `0.3` -> `0.2`
  - `infer.forest_topk_turn_penalty`: `0.3` -> `0.2`
  - `train.forest_min_progress_m`: `0.0` -> `0.02`
  - `infer.forest_min_progress_m`: `0.0` -> `0.02`
  - `train.forest_train_short_prob`: `0.25` -> `0.20`
  - `train.forest_train_dynamic_target_sr_long`: `0.85` -> `0.90`
  - `train.forest_train_dynamic_min_short_prob`: `0.25` -> `0.20`
  - `train.forest_demo_pretrain_steps`: `12000` -> `15000`
  - `train.forest_demo_pretrain_min_effective_steps`: `3000` -> `4000`
  - `train.forest_demo_pretrain_val_runs`: `6` -> `8`
  - `train.replay_capacity`: `8000` -> `10000`

### 2) 可复现配置
- `configs/repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke.json`（新增）
  - 作为 `v7p3p7` 的可复现实验入口（train+infer 同口径）。

### 3) 版本留档
- `docs/versions/v7p3p7/README.md`（新增）
- `docs/versions/v7p3p7/CHANGES.md`（新增）
- `docs/versions/v7p3p7/RESULTS.md`（新增）
- `docs/versions/v7p3p7/runs/README.md`（新增）

### 4) 索引/总览同步
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
- 已新增 `v7p3p7` 版本索引条目并回填 smoke 结果与状态。

## 代码变更说明
- 本版无 Python 代码逻辑改动，仅新增配置与版本文档。
