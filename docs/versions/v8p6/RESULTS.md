# v8p6 结果对比（待回填）

> 说明：本版新增 `--forest-replace-topq`（替换动作候选 Top-Q 约束），默认不启用（`0`）。回填结果时必须写清 `forest_replace_ranking` 与 `forest_replace_topq` 的取值。

## 1) 关键工件路径

- infer-only smoke（固定 `v7p1` checkpoint，runs=3）：
  - profile：`configs/repro_20260223_v8p6_replace_topq_infer_smoke.json`
  - run_dir：`N/A`
- train+infer smoke（episodes=150, runs=3）：
  - profile：`configs/v8p6.json`
  - train_run：`N/A`
  - infer_run：`N/A`

## 2) short/mid/long KPI（infer-only smoke）

- `table2_kpis_mean_raw.csv`：`N/A`
- `failure_reason` 分布（来自 `table2_kpis_raw.csv`）：`N/A`

## 3) short/mid/long KPI（train+infer smoke）

- `table2_kpis_mean_raw.csv`：`N/A`
- `failure_reason` 分布：`N/A`

## 4) 门槛检查（最终门槛仅供格式，未评测）

- short（runs=20）：`N/A`
- long（runs=20）：`N/A`

## 5) 结论（待回填）

- `N/A`

