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

## 2026-02-11 三次追加（mid=14–42m 训练+测试接入）
- 新增 mid 固定 pairs 配置（20 组）：
  - `configs/repro_20260211_forest_a_pairs_mid20_v1.json`
- 新增 mid 推理复评配置（strict/hybrid，runs=20）：
  - `configs/repro_20260211_v5_reval_v3p11_strict_mid_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_hybrid_mid_pairs20_v1.json`
- 新增 mid 覆盖训练配置（两套件保持不变，long 起点下限降到 14m）：
  - `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1.json`
- 新增 mid 运行留档：
  - pairgen：`runs/repro_20260211_pairgen_forest_a_mid20/20260211_153822`
  - strict smoke：`runs/repro_20260211_v5_smoke_strict_mid_pairs5_v1/20260211_154838`
  - hybrid smoke：`runs/repro_20260211_v5_smoke_hybrid_mid_pairs5_v1/20260211_154851`
- 同轮更新：
  - `docs/versions/v5/README.md`
  - `docs/versions/v5/RESULTS.md`
  - `docs/versions/v5/runs/README.md`
  - `README.md`
  - `README.zh-CN.md`

## 2026-02-11 四次追加（train=300 + infer=20 效果复测）
- 执行 300 轮训练（使用 midcover 配置，保留 early-stop 机制）：
  - command：`conda run -n ros2py310 python train.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1 --episodes 300 --out repro_20260211_v5_train300_midcover_v1`
  - run：`runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840`
  - 训练停止：`episodes_completed=150/300`，`stop_reason=rl_early_stop_plateau`
- 基于新模型执行 infer runs=20（mid fixed pairs20）：
  - command：`conda run -n ros2py310 python infer.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1 --models runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models --runs 20 --out repro_20260211_v5_train300_midcover_infer20_v1`
  - run：`runs/repro_20260211_v5_train300_midcover_infer20_v1/20260211_164304`
- 指标摘要（mid）：
  - `CNN-DDQN`：SR=`0.80`，path_time_s=`16.9813`，argmax_inad=`0.147`，failure=`reached=16, timeout=4`
  - `Hybrid A*-MPC`：SR=`0.95`，path_time_s=`12.6553`，failure=`reached=19, collision=1`
- 同轮更新：
  - `docs/versions/v5/README.md`
  - `docs/versions/v5/CHANGES.md`
  - `docs/versions/v5/RESULTS.md`
  - `docs/versions/v5/runs/README.md`

## 2026-02-11 五次追加（补齐 short/long runs=20）
- 在同一新模型（`runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models`）下补齐：
  - short command：`conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1 --models runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models --baselines hybrid_astar_mpc --out repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1`
  - long command：`conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1 --models runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models --baselines hybrid_astar_mpc --out repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1`
- 新增 run：
  - short：`runs/repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1/20260211_172605`
  - long：`runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706`
- 三套件结果（CNN-DDQN SR）：`short=0.75, mid=0.80, long=0.65`。
- 同轮更新：
  - `docs/versions/v5/README.md`
  - `docs/versions/v5/CHANGES.md`
  - `docs/versions/v5/RESULTS.md`
  - `docs/versions/v5/runs/README.md`

## 参数与口径快照（重点）
- 主模型 checkpoint：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models`
- strict 口径：`forest_no_fallback=true`
- hybrid 口径：`forest_no_fallback=false`
- 评测样本：fixed pairs20 v1
  - short：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - mid：`configs/repro_20260211_forest_a_pairs_mid20_v1.json`
  - long：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
- 公平设置：`runs=20`、`random_start_goal=true`、`rand_two_suites=false`
- 四基线同场设置：`--baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc`
- mid 距离窗口：`rand_min_dist_m=14.0`、`rand_max_dist_m=42.0`

## 变更文件清单
- `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke.json`
- `configs/repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1.json`
- `configs/repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1.json`
- `configs/repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1.json`
- `configs/repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1.json`
- `configs/repro_20260211_forest_a_pairs_mid20_v1.json`
- `configs/repro_20260211_v5_reval_v3p11_strict_mid_pairs20_v1.json`
- `configs/repro_20260211_v5_reval_v3p11_hybrid_mid_pairs20_v1.json`
- `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1.json`
- `docs/versions/v5/README.md`
- `docs/versions/v5/CHANGES.md`
- `docs/versions/v5/RESULTS.md`
- `docs/versions/v5/runs/README.md`
- `README.md`
- `README.zh-CN.md`
- `docs/versions/README.md`
