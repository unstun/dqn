# v8p9 变更清单（相对 v8p8）

## 1) 主要策略（推理优先）

- 本版优先做“推理侧 sweep”：在不改到达判定阈值的前提下，通过调 `forest_min_progress_m/forest_min_od_m/forest_replace_topq/forest_topk_turn_penalty/forest_goal_approach_speed_factor` 尝试压 `avg_path_length/path_time_s`。
- 训练侧暂不作为第一优先；待推理侧上限确认后再决定是否进入训练侧（reward / DQfD 约束再平衡等）。

## 2) 配置与文档

- 新增 `configs/v8p9.json`（版本入口：带默认推理候选参数）
- 新增 `configs/repro_20260224_v8p9_infer_sweep_{short,long}_smoke.json`（推理侧 sweep smoke，可复现）
- 新增 `configs/pairs_v8p9_smoke_{short,long}3_from_pairs20_v1_20260224.json`（从 pairs20 抽取的 pairs3 子集）
- 新增 `docs/versions/v8p9/` 四件套（本文件为变更明细）

## 3) 受影响文件清单

- `configs/v8p9.json`
- `configs/repro_20260224_v8p9_infer_sweep_short_smoke.json`
- `configs/repro_20260224_v8p9_infer_sweep_long_smoke.json`
- `configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
- `configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
- `docs/versions/v8p9/README.md`
- `docs/versions/v8p9/CHANGES.md`
- `docs/versions/v8p9/RESULTS.md`
- `docs/versions/v8p9/runs/README.md`

