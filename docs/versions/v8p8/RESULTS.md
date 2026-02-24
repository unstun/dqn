# v8p8 结果对比（smoke 已跑；full gate C 待跑）

> 说明：本版目标是最终硬门槛 C（short/long 各 runs=20，fixed pairs20）。在 full gate 结果出来前，所有 smoke 仅用于筛查，不作为最终结论。

## 1) 关键工件路径

- smoke（episodes=150, runs=3）：
  - profile：`configs/repro_20260224_v8p8_smoke.json`
  - train_run：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059`
  - infer_run：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556`
  - kpi_mean_raw：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556/table2_kpis_mean_raw.csv`
  - kpi_raw：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556/table2_kpis_raw.csv`

- 推理侧消融（固定 pairs3，runs=3；避免 sample drift）：
  - short pairs：`configs/pairs_v8p8_smoke_short3_20260224_110556.json`
  - long pairs：`configs/pairs_v8p8_smoke_long3_20260224_110556.json`
  - short r=1.5：`runs/v8p8_ablate_pairs3_short_r15_sf0p9/20260224_111755/table2_kpis_mean_raw.csv`
  - short r=2.5：`runs/v8p8_ablate_pairs3_short_r25_sf0p9/20260224_111838/table2_kpis_mean_raw.csv`
  - long r=1.5：`runs/v8p8_ablate_pairs3_long_r15_sf0p9/20260224_111905/table2_kpis_mean_raw.csv`
  - long r=2.5：`runs/v8p8_ablate_pairs3_long_r25_sf0p9/20260224_111932/table2_kpis_mean_raw.csv`

- full gate（C：fixed pairs20；short/long 各 runs=20）：
  - short pairs：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - long pairs：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
  - short run_dir：`N/A`
  - long run_dir：`N/A`

## 2) smoke 结果（runs=3，mean）

来自：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556/table2_kpis_mean_raw.csv`

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate | fallback_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 17.0527 | 10.1250 | 0.161333 | 0.107 | 0.107 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | N/A | N/A |
| mid | CNN-DDQN | 1.000 | 30.4747 | 17.6000 | 0.227513 | 0.194 | 0.194 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | N/A | N/A |
| long | CNN-DDQN | 1.000 | 53.9671 | 29.3500 | 0.197854 | 0.204 | 0.204 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | N/A | N/A |

`failure_reason` 分布（CNN-DDQN，runs=3）：
- short：`reached=2`，`timeout=1`
- mid：`reached=3`
- long：`reached=3`

## 3) full gate（C）结果（runs=20，fixed pairs）

- short：`N/A`
- long：`N/A`

门槛检查（C）：
- `N/A`（等待填入 short/long 的 `table2_kpis_mean_raw.csv` 后判定）

## 4) 推理侧消融（固定 pairs3，runs=3）

### 4.1 `forest_goal_admissible_relax_factor`：1.5 vs 2.5（short/long）

结论：在该 3 对固定样本上，`relax_factor` 从 1.5 调到 2.5 对 CNN-DDQN 指标无可见影响（`success_rate/avg_path_length/path_time_s` 基本一致；`fallback_rate` 也一致）。

备注（公平性风险提示）：
- 该组 short 固定 pairs3 上，Hybrid A*-MPC 在 `run_idx=2` 出现一次 `collision`，导致 baseline `success_rate=0.667`。后续 gate C 仍以 pairs20 + runs=20 的稳定口径为准，不以该 3 对样本作最终结论。
