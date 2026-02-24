# v8p9 结果对比（推理侧 sweep smoke 已跑；full gate C 待跑）

> 说明：本版目标是最终硬门槛 C（short/long 各 runs=20，fixed pairs20）。在 full gate 结果出来前，所有 smoke 仅用于筛查，不作为最终结论。

## 1) 关键工件路径

- 推理侧 sweep smoke（fixed pairs3，runs=3）：
  - short profile：`configs/repro_20260224_v8p9_infer_sweep_short_smoke.json`
  - long profile：`configs/repro_20260224_v8p9_infer_sweep_long_smoke.json`
  - short pairs：`configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
  - long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
  - short run_dir（默认候选）：`runs/v8p9_infer_sweep_short_pairs3_smoke/20260224_114743`
  - long run_dir（默认候选）：`runs/v8p9_infer_sweep_long_pairs3_smoke/20260224_114743`

- full gate（C：fixed pairs20；short/long 各 runs=20）：
  - short pairs：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - long pairs：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
  - short run_dir：`N/A`
  - long run_dir：`N/A`

## 2) sweep smoke 结果（runs=3，mean）

固定 pairs3 的 baseline（Hybrid A*-MPC）在该样本集上显著短于 RL（这是我们需要追平/反超的目标）。

### 2.1 short（fixed pairs3）

baseline（Hybrid A*-MPC）：`avg_path_length=16.3207`，`path_time_s=9.4667`，`success_rate=1.0`

| 变体 | RL success_rate | RL avg_path_length | RL path_time_s | 备注 |
|---|---:|---:|---:|---|
| 默认候选（profile 默认） | 0.667 | 20.7807 | 12.2250 | `runs/v8p9_infer_sweep_short_pairs3_smoke/20260224_114743` |
| varA（v8p8-like） | **1.000** | **17.4091** | **10.9500** | `runs/v8p9_sweep_short_varA_v8p8like/20260224_114851` |
| varB（od=0） | 1.000 | 20.9982 | 12.6333 | `runs/v8p9_sweep_short_varB_od0/20260224_115005` |
| varC（q-rank，topq=0） | 0.667 | 21.7426 | 12.5000 | `runs/v8p9_sweep_short_varC_qrank/20260224_115110`（SR 退化） |
| varD（mp=0.02, od=0） | 1.000 | 20.9599 | 12.6167 | `runs/v8p9_sweep_short_varD_mp0p02_od0/20260224_115140` |

### 2.2 long（fixed pairs3）

baseline（Hybrid A*-MPC）：`avg_path_length=32.2801`，`path_time_s=17.4333`，`success_rate=1.0`

| 变体 | RL success_rate | RL avg_path_length | RL path_time_s | 备注 |
|---|---:|---:|---:|---|
| 默认候选（profile 默认） | 1.000 | 55.1412 | 30.1833 | `runs/v8p9_infer_sweep_long_pairs3_smoke/20260224_114743` |
| varA（v8p8-like） | **1.000** | 53.6843 | 28.6000 | `runs/v8p9_sweep_long_varA_v8p8like/20260224_114852` |
| varB（od=0） | **1.000** | **52.1466** | **27.7667** | `runs/v8p9_sweep_long_varB_od0/20260224_115006`（当前 best） |
| varC（q-rank，topq=0） | 0.667 | 62.0682 | 32.8750 | `runs/v8p9_sweep_long_varC_qrank/20260224_115110`（SR 退化） |
| varD（mp=0.02, od=0） | 1.000 | 56.2472 | 30.0000 | `runs/v8p9_sweep_long_varD_mp0p02_od0/20260224_115141` |

**阶段结论（smoke）**：
- 在该 fixed pairs3 子集上，推理侧调参可以把 SR 拉回到 1.0（short/long），并小幅压缩 RL 的 path/time，但 **仍显著落后 Hybrid A*-MPC**（尤其 long）。
- 因此 v8p9 下一步不建议直接上 full gate C；更合理的是：在保持 varA/varB 方向的前提下，进入训练侧或更强 DQN 变种（否则很难弥合 long 上 ~20m+ 的路径差距）。

## 3) full gate（C）结果（runs=20，fixed pairs）

- short：`N/A`
- long：`N/A`

门槛检查（C）：
- `N/A`（等待填入 short/long 的 `table2_kpis_mean_raw.csv` 后判定）
