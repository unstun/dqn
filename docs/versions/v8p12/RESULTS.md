# v8p12 结果对比（待跑：train+infer smoke → 决定是否 full gate C）

> 说明：本版目标是最终硬门槛 C（short/long 各 runs=20，fixed pairs20）。在 full gate 结果出来前，所有 smoke 仅用于筛查，不作为最终结论。

## 1) 关键工件路径（计划）

- train smoke（episodes=150）：
  - train profile：`configs/repro_20260224_v8p12_train_smoke.json`
  - train run_dir：`runs/v8p12/train_20260224_162127`

- infer smoke（fixed pairs3，runs=3）：
  - short profile：`configs/repro_20260224_v8p12_infer_smoke_short.json`
  - long profile：`configs/repro_20260224_v8p12_infer_smoke_long.json`
  - short pairs：`configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
  - long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
  - short run_dir：`runs/v8p12_infer_smoke_short_pairs3/20260224_163604`
  - long run_dir：`runs/v8p12_infer_smoke_long_pairs3/20260224_163613`

- full gate（C：fixed pairs20；short/long 各 runs=20）：
  - short pairs：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - long pairs：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
  - short run_dir：`N/A`
  - long run_dir：`N/A`

## 2) smoke 结果（runs=3 / episodes=150，mean）

> KPI 来源（每个 run_dir 的 `table2_kpis_mean_raw.csv`）。

### short（fixed pairs3，runs=3）

- KPI：`runs/v8p12_infer_smoke_short_pairs3/20260224_163604/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=18.7002`，`path_time_s=11.2667`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=16.3207`，`path_time_s=9.4667`
- Δ（RL - baseline）：`avg_path_length=+2.3795`，`path_time_s=+1.8000`（short 明显回退）

### long（fixed pairs3，runs=3）

- KPI：`runs/v8p12_infer_smoke_long_pairs3/20260224_163613/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=37.3957`，`path_time_s=20.5833`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=32.2801`，`path_time_s=17.4333`
- Δ（RL - baseline）：`avg_path_length=+5.1156`，`path_time_s=+3.1500`（long 仍落后）

结论（smoke 口径）：
- long 的 detour 有小幅回落（相对 v8p11 约缩短 ~0.72m），但仍不足以打败 baseline；同时 short 出现明显回退。
- 本版判定：**NO-GO（暂不建议 full gate C）**。

## 3) full gate（C）结果（runs=20，fixed pairs）

- short：`N/A`
- long：`N/A`

门槛检查（C）：
- `N/A`（等待填入 short/long 的 `table2_kpis_mean_raw.csv` 后判定）
