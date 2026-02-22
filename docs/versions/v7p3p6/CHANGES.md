# v7p3p6 改动清单（相对 v7p3p4）

## 变更目标
- 在不改算法定义与网络模块的前提下，针对 `obs_map_size=128` 下的 long 超时问题做参数纠偏（smoke 优先）。

## 配置改动明细

### 1) 主版本配置
- `configs/v7p3p6.json`（新增）
  - 基于 `v7p3p4` 重建并改名为 `v7p3p6`
  - `train.out`: `v7p3p4` -> `v7p3p6_obsmap128_tune_smoke`
  - `infer.models/out`: `v7p3p4` -> `v7p3p6_obsmap128_tune_smoke`
  - `train.episodes`: `300` -> `150`
  - `infer.runs`: `5` -> `3`
  - `train.replay_capacity`: `<unset>` -> `8000`
  - `train.batch_size`: `128` -> `32`
  - `train.save_ckpt_suite_runs`: `10` -> `3`
  - `train.obs_map_size`: `<unset>` -> `128`
  - `infer.obs_map_size`: `<unset>` -> `128`
  - `train.forest_topk_turn_penalty`: `1.0` -> `0.3`
  - `infer.forest_topk_turn_penalty`: `1.0` -> `0.3`
  - `train.forest_min_progress_m`: `-0.01` -> `0.0`
  - `infer.forest_min_progress_m`: `-0.01` -> `0.0`
  - `train.forest_train_short_prob`: `0.35` -> `0.25`
  - `train.forest_train_dynamic_target_sr_long`: `0.75` -> `0.85`
  - `train.forest_demo_target_mult`: `<unset>` -> `10.0`
  - `train.forest_demo_target_cap`: `<unset>` -> `8000`
  - `train.forest_demo_pretrain_steps`: `30000` -> `12000`
  - `train.forest_demo_pretrain_min_effective_steps`: `8000` -> `3000`
  - `train.forest_demo_pretrain_val_runs`: `<unset>` -> `6`

### 2) 可复现配置（新增）
- `configs/repro_20260222_v7p3p6_obsmap128_tune_smoke.json`
  - 作为 `v7p3p6` 的可复现实验入口（train+infer 同口径）。

### 3) 版本留档（新增）
- `docs/versions/v7p3p6/README.md`
- `docs/versions/v7p3p6/CHANGES.md`
- `docs/versions/v7p3p6/RESULTS.md`
- `docs/versions/v7p3p6/runs/README.md`

### 4) 索引/总览同步
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
- `v7p3p6` 新增到版本总索引，并回填 smoke 结果与状态。

## 代码变更说明
- 本版无 Python 代码逻辑改动，仅新增配置与版本文档。
