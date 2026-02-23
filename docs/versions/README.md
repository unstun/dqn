# 版本留档索引（v1 → v8p6，含 v7p1 补档）

- 归档入口：仓库根目录 `README.md` 的“版本总索引（v1 → v8p6）”。
- 本文件保留为 `docs/versions/` 内部快速导航，与根 README 保持同一口径。
- 历史目录 `v3p1`~`v3p11` 保留原始记录，未纳入本轮重编号。
- 早期误混入版本链已于 2026-02-09 清理，当前主线编号延续至 `v8p6`。
- 当前主线对外口径（`v6p2p3` 及后续）统一为 `CNN-DDQN (shielded/hybrid inference)`；`strict-argmax` 仅用于诊断/消融，不作为主结论口径。
- `v7p1` 已按版本四件套补档，作为当前稳定主线归档入口。

| 版本 | 历史来源 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|---|
| `v1` | `v1` | `docs/versions/v1/` | `configs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke.json` | `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017` | `0.0` / `0.0` | `1.0` / `1.0` | 未通过 |
| `v2` | `v2` | `docs/versions/v2/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246` | `0.0` / `0.0` | `1.0` / `1.0` | 未通过 |
| `v3` | `v3` | `docs/versions/v3/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403` | `0.5` / `0.1` | `0.9` / `1.0` | 未通过 |

## 增量版本（v3p1 → v8p6）

| 版本 | 历史来源 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|---|
| `v3p12` | `v3p12` | `docs/versions/v3p12/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622` | `0.0` / `0.0` | `0.95` / `1.0` | 未通过 |
| `v4p1` | `v4p1` | `docs/versions/v4p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10/20260210_135524` | `0.1` / `0.0` | `0.9` / `1.0` | 未通过 |
| `v4p2` | `v4p2` | `docs/versions/v4p2/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730` | `0.0` / `0.0` | `0.9` / `1.0` | 未通过 |
| `v4p3` | `v4p3` | `docs/versions/v4p3/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934` | `0.2` / `0.0` | `0.9` / `1.0` | 未通过 |
| `v4p3p1` | `v4p3p1` | `docs/versions/v4p3p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044` | `0.0` / `0.0` | `0.9` / `1.0` | 未通过 |
| `v5` | `v5` | `docs/versions/v5/` | `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1.json` | `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351` | `0.75` / `0.85` | `0.95` / `0.90` | 未通过 |
| `v6` | `v6` | `docs/versions/v6/` | `configs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602` | `0.90` / `0.70` | `0.95` / `0.90` | 未通过 |
| `v6p1` | `v6p1` | `docs/versions/v6p1/` | `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414` | `0.70` / `0.95` | `0.95` / `0.90` | 未通过 |
| `v6p2` | `v6p2` | `docs/versions/v6p2/` | `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414` | `0.70` / `0.95` | `0.95` / `0.90` | 未通过 |
| `v6p2p2` | `v6p2p2` | `docs/versions/v6p2p2/` | `configs/v6p2p2.json` | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433` | `0.75` / `0.55` | `0.95` / `1.00` | 未通过（待 full） |
| `v6p2p3` | `v6p2p3` | `docs/versions/v6p2p3/` | `configs/v6p2p3.json` | `runs/v6p2p3/train_20260219_142104/infer/20260219_145315` | `0.80` / `1.00` | `1.00` / `1.00` | 已运行（runs=5，待 full20） |
| `v7p1` | `v7p1` | `docs/versions/v7p1/` | `configs/v7p1.json` | `runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927` | `1.00` / `1.00` | `1.00` / `1.00` | 稳定主线（runs=5，待 full20） |
| `v7p2` | `v7p2` | `docs/versions/v7p2/` | `configs/v7p2.json` | `runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137` | `1.00` / `1.00` | `1.00` / `1.00` | 已运行（smoke/micro-smoke：episodes=40, runs=3） |
| `v7p2p1` | `v7p2p1` | `docs/versions/v7p2p1/` | `configs/repro_20260220_v7p2p1_rollback_v7p1.json` | `runs/v7p2_es150/train_20260220_222056/infer/20260220_223016` | `0.85` / `0.65` | `0.95` / `1.00` | 失败归档，主线回退到 `v7p1` |
| `v7p2p2` | `v7p2p2` | `docs/versions/v7p2p2/` | `configs/repro_20260221_v7p2p2_globalcnn_smoke.json` | `runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943` | `0.667` / `0.333` | `1.00` / `1.00` | 失败归档（smoke 不达门，主线保持 `v7p1`） |
| `v7p2p3` | `v7p2p3` | `docs/versions/v7p2p3/` | `configs/repro_20260221_v7p2p3_globalcnn_fusion_smoke.json` | `runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334/infer/20260221_180256` | `0.333` / `0.667` | `1.00` / `1.00` | 失败归档（smoke 不达门，主线保持 `v7p1`） |
| `v7p2p4` | `v7p2p4` | `docs/versions/v7p2p4/` | `configs/repro_20260221_v7p2p4_globalcnn_spatialprior_smoke.json` | `runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926` | `0.667` / `1.000` | `1.00` / `1.00` | 失败归档（smoke 不达门，保持当前代码并继续前向迭代） |
| `v7p2p5` | `v7p2p5` | `docs/versions/v7p2p5/` | `configs/repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke.json` | `runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626` | `0.333` / `0.667` | `1.00` / `1.00` | 失败归档（smoke 退化，不回退代码并继续前向迭代） |
| `v7p2p6` | `v7p2p6` | `docs/versions/v7p2p6/` | `configs/repro_20260221_v7p2p6_foundationfix_smoke.json` | `runs/v7p2p6_foundationfix_smoke/train_20260221_211603/infer/20260221_213248` | `1.000` / `0.000` | `1.00` / `1.00` | 失败归档（short 改善但 long 崩塌，继续前向迭代） |
| `v7p2p7` | `v7p2p7` | `docs/versions/v7p2p7/` | `configs/repro_20260221_v7p2p7_gradclip_recover_smoke.json` | `runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452/infer/20260221_221008` | `0.333` / `0.333` | `1.00` / `1.00` | 失败归档（long 有恢复但 short 退化，继续前向迭代） |
| `v7p2p8` | `v7p2p8` | `docs/versions/v7p2p8/` | `configs/repro_20260221_v7p2p8_bold_dynamic_expert_smoke.json` | `runs/v7p2p8_bold_dynamic_expert_smoke/train_20260221_225358/infer/20260221_230426` | `0.000` / `1.000` | `1.00` / `1.00` | 失败归档（long 恢复到 1.0，但 short 崩塌到 0.0，继续前向迭代） |
| `v7p2p9` | `v7p2p9` | `docs/versions/v7p2p9/` | `configs/repro_20260221_v7p2p9_ablate_expert_smoke.json` | `runs/v7p2p9_ablate_expert_smoke/train_20260221_231402/infer/20260221_232825` | `0.667` / `0.000` | `1.00` / `1.00` | 失败归档（short 回升但 long 崩塌，继续前向迭代） |
| `v7p2p10` | `v7p2p10` | `docs/versions/v7p2p10/` | `configs/repro_20260221_v7p2p10_penalty035_smoke.json` | `runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340` | `0.667` / `0.333` | `1.00` / `1.00` | 失败归档（long 回升但 short 路径与平滑性退化，继续前向迭代） |
| `v7p3` | `v7p3` | `docs/versions/v7p3/` | `configs/repro_20260221_v7p3_suite_penalty_smoke.json` | `runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023` | `0.667` / `0.333` | `1.00` / `1.00` | 失败归档（short/mid 局部改善但 long path/time 退化，未过 smoke 门） |
| `v7p3p1` | `v7p3p1` | `docs/versions/v7p3p1/` | `configs/repro_20260222_v7p3p1_adaptive_penalty_smoke.json` | `runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552` | `0.667` / `1.000` | `1.00` / `1.00` | 失败归档（mid/long SR 提升至 1.0，但 path/time/smoothness 全面退化） |
| `v7p3p2` | `v7p3p2` | `docs/versions/v7p3p2/` | `configs/repro_20260222_v7p3p2_turnaware_smoke.json` | `runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842` | `0.333` / `0.333` | `1.00` / `1.00` | 失败归档（路径/时间有所回落，但三套件 SR 显著下降，未过 smoke 门） |
| `v7p3p3` | `v7p3p3` | `docs/versions/v7p3p3/` | `configs/repro_20260222_v7p3p3_infergate_smoke.json` | `runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657` | `0.000` / `0.667` | `1.00` / `1.00` | 失败归档（long SR 回升，但 short=0 且出现碰撞/超时，未过 smoke 门） |
| `v7p3p4` | `v7p3p4` | `docs/versions/v7p3p4/` | `configs/repro_20260222_v7p3p4_safe_fallback_infer_smoke.json` | `runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513` | `0.667` / `1.000` | `1.00` / `1.00` | 失败归档（safe fallback 补丁修复碰撞回潮；但 short/mid SR 仍落后 baseline，且 path/time 更差；本轮为 infer-only smoke） |
| `v7p3p6` | `v7p3p6` | `docs/versions/v7p3p6/` | `configs/repro_20260222_v7p3p6_obsmap128_tune_smoke.json` | `runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007/infer/20260222_223831` | `0.667` / `0.333` | `1.00` / `1.00` | 失败归档（long 从 0.000 回升到 0.333，但 short/long 仍未过门） |
| `v7p3p7` | `v7p3p7` | `docs/versions/v7p3p7/` | `configs/repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke.json` | `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329` | `1.000` / `0.333` | `1.00` / `1.00` | 失败归档（short/mid SR 升至 1.0 且 CNN 总 timeout 从 5 降到 2；但 long 仍 2/3 timeout，short/long path-time 仍落后 baseline） |
| `v8p1` | `v8p1` | `docs/versions/v8p1/` | `configs/v8p1.json` | `runs/v8p1_navdist_smoke/train_20260223_021339/infer/20260223_023932` | `0.667` / `0.333` | `1.00` / `1.00` | 失败归档（navdist progress distance；smoke SR 退化） |
| `v8p2` | `v8p2` | `docs/versions/v8p2/` | `configs/v8p2.json` | `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027` | `0.667` / `1.000` | `1.00` / `1.00` | smoke 已跑（mid/long=1.0；short=2/3 collision；暂不 full） |
| `v8p3` | `v8p3` | `docs/versions/v8p3/` | `configs/v8p3.json` | `runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153` | `1.000` / `0.667` | `1.00` / `1.00` | 失败归档（smoke：mid collision=1/3；long timeout=1/3） |
| `v8p4` | `v8p4` | `docs/versions/v8p4/` | `configs/v8p4.json` | `runs/v8p4_smoke_failures_regression/20260223_142739` | `N/A` / `N/A` | `N/A` / `N/A` | 失败归档（回归：mid/long 各 collision=1/2、timeout=1/2；暂不 smoke） |
| `v8p5` | `v8p5` | `docs/versions/v8p5/` | `configs/v8p5.json` | `runs/v8p5_replace_ranking_infer_smoke/20260223_172217` | `1.00` / `1.00` | `1.00` / `1.00` | infer-only smoke：`q` PASS；tie-break short `collision=1/3`（NO-GO）；train+infer smoke 未跑 |
| `v8p6` | `v8p6` | `docs/versions/v8p6/` | `configs/v8p6.json` | `runs/v8p6_replace_topq_infer_smoke/20260223_185628` | `1.00` / `1.00` | `1.00` / `1.00` | infer-only smoke 通过（topq=1/2/3；推荐 topq=3）；train+infer smoke 未跑 |

## baseline-only 排除口径
- 上表“关键 run”仅统计 RL 运行（`skip_rl=false`）。
- baseline-only（`--skip-rl`）输出请单独查看 `runs/outputs_forest_baselines/*` 及 `runs/repro_20260207_*` 系列。
