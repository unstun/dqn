# v8p11 结果对比（待跑：train+infer smoke → full gate C）

> 说明：本版目标是最终硬门槛 C（short/long 各 runs=20，fixed pairs20）。在 full gate 结果出来前，所有 smoke 仅用于筛查，不作为最终结论。

## 1) 关键工件路径（计划）

- train smoke（episodes=150）：
  - train profile：`configs/repro_20260224_v8p11_train_smoke.json`
  - train run_dir：`N/A`

- infer smoke（fixed pairs3，runs=3）：
  - short profile：`configs/repro_20260224_v8p11_infer_smoke_short.json`
  - long profile：`configs/repro_20260224_v8p11_infer_smoke_long.json`
  - short pairs：`configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
  - long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
  - short run_dir：`N/A`
  - long run_dir：`N/A`

- full gate（C：fixed pairs20；short/long 各 runs=20）：
  - short pairs：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - long pairs：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
  - short run_dir：`N/A`
  - long run_dir：`N/A`

## 2) smoke 结果（runs=3 / episodes=150，mean）

- short：`N/A`
- long：`N/A`

## 3) full gate（C）结果（runs=20，fixed pairs）

- short：`N/A`
- long：`N/A`

门槛检查（C）：
- `N/A`（等待填入 short/long 的 `table2_kpis_mean_raw.csv` 后判定）

