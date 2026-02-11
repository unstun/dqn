# v5 - 变更

## 版本意图
- 将历史效果更优的 `v3p11` 模型口径提升为新版本主评测基线，并规范 strict/hybrid 双口径复评测流程。

## 相对 v4p3p1 的具体变更
- 代码实现：`无`（未修改 `forest_vehicle_dqn/**`）。
- 新增 v5 可复现配置：
  - `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke.json`
  - `configs/repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1.json`
- 新增版本留档：
  - `docs/versions/v5/README.md`
  - `docs/versions/v5/CHANGES.md`
  - `docs/versions/v5/RESULTS.md`
  - `docs/versions/v5/runs/README.md`
- 索引同步：
  - 根目录 `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`

## 2026-02-11 追加（baseline 对比）
- 新增 baseline-only fixed pairs20 运行（Hybrid A*-MPC）：
  - short：`runs/repro_20260211_v5_baseline_hybrid_short_pairs20_v1/20260211_090612`
  - long：`runs/repro_20260211_v5_baseline_hybrid_long_pairs20_v1/20260211_090655`
- 更新 `docs/versions/v5/README.md`、`docs/versions/v5/RESULTS.md`、`docs/versions/v5/runs/README.md` 的 baseline 指标与路径。

## 2026-02-11 二次追加（四基线同场对照）
- 新增 RL 同场四基线对照运行（`Hybrid A*` + `Hybrid A*-MPC` + `RRT*` + `RRT-MPC`）：
  - strict short：`runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538`
  - strict long：`runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712`
  - hybrid short：`runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220`
  - hybrid long：`runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351`
- 统一后续 v5 baseline 对照口径为四基线同场（不再仅看 `Hybrid A*-MPC` 单基线）。
- 同轮更新：
  - `docs/versions/v5/README.md`
  - `docs/versions/v5/RESULTS.md`
  - `docs/versions/v5/runs/README.md`

## 参数与口径快照（重点）
- 主模型 checkpoint：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models`
- strict 口径：`forest_no_fallback=true`
- hybrid 口径：`forest_no_fallback=false`
- 评测样本：fixed pairs20 v1
  - short：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - long：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
- 公平设置：`runs=20`、`random_start_goal=true`、`rand_two_suites=false`
- 四基线同场设置：`--baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc`

## 变更文件清单
- `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke.json`
- `configs/repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1.json`
- `configs/repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1.json`
- `configs/repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1.json`
- `configs/repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1.json`
- `docs/versions/v5/README.md`
- `docs/versions/v5/CHANGES.md`
- `docs/versions/v5/RESULTS.md`
- `docs/versions/v5/runs/README.md`
- `README.md`
- `README.zh-CN.md`
- `docs/versions/README.md`
