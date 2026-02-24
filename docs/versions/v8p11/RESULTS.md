# v8p11 结果对比（待跑：train+infer smoke → full gate C）

> 说明：本版目标是最终硬门槛 C（short/long 各 runs=20，fixed pairs20）。在 full gate 结果出来前，所有 smoke 仅用于筛查，不作为最终结论。

## 1) 关键工件路径（计划）

- train smoke（episodes=150）：
  - train profile：`configs/repro_20260224_v8p11_train_smoke.json`
  - train run_dir：`runs/v8p11/train_20260224_151042`

- infer smoke（fixed pairs3，runs=3）：
  - short profile：`configs/repro_20260224_v8p11_infer_smoke_short.json`
  - long profile：`configs/repro_20260224_v8p11_infer_smoke_long.json`
  - short pairs：`configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
  - long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
  - short run_dir：`runs/v8p11_infer_smoke_short_pairs3/20260224_152858`
  - long run_dir：`runs/v8p11_infer_smoke_long_pairs3/20260224_152917`

- full gate（C：fixed pairs20；short/long 各 runs=20）：
  - short pairs：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - long pairs：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
  - short run_dir：`N/A`
  - long run_dir：`N/A`

## 2) smoke 结果（runs=3 / episodes=150，mean）

> KPI 来源（每个 run_dir 的 `table2_kpis_mean_raw.csv`）。

### short（fixed pairs3，runs=3）

- KPI：`runs/v8p11_infer_smoke_short_pairs3/20260224_152858/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=15.067`，`path_time_s=9.1333`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=16.3207`，`path_time_s=9.4667`
- Δ（RL - baseline）：`avg_path_length=-1.2537`，`path_time_s=-0.3334`（short 上已满足“更短/更快”）

### long（fixed pairs3，runs=3）

- KPI：`runs/v8p11_infer_smoke_long_pairs3/20260224_152917/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=38.113`，`path_time_s=20.5167`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=32.2801`，`path_time_s=17.4333`
- Δ（RL - baseline）：`avg_path_length=+5.8329`，`path_time_s=+3.0834`（long 仍落后 baseline）

结论（smoke 口径）：
- short：SR=1.0 且路径/时间均优于 baseline（达成）
- long：SR=1.0，但路径/时间仍劣于 baseline（未达成；暂不建议直接上 full gate C）

## 3) full gate（C）结果（runs=20，fixed pairs）

- short：`N/A`
- long：`N/A`

门槛检查（C）：
- `N/A`（等待填入 short/long 的 `table2_kpis_mean_raw.csv` 后判定）
