# v8p6 runs 追溯

## 1) 代表 run（最小追溯字段）

- infer-only smoke（固定 `v7p1` checkpoint；同一随机对；runs=3；`forest_replace_ranking=progress_clearance_q`）
  - topq=2（默认）
    - run_dir：`runs/v8p6_replace_topq_infer_smoke/20260223_185519`
    - run_json：`runs/v8p6_replace_topq_infer_smoke/20260223_185519/configs/run.json`
    - kpi_mean_raw：`runs/v8p6_replace_topq_infer_smoke/20260223_185519/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p6_replace_topq_infer_smoke/20260223_185519/table2_kpis_raw.csv`
  - topq=1（≈纯 Q replacement 对照）
    - run_dir：`runs/v8p6_replace_topq_infer_smoke/20260223_185553`
    - run_json：`runs/v8p6_replace_topq_infer_smoke/20260223_185553/configs/run.json`
    - kpi_mean_raw：`runs/v8p6_replace_topq_infer_smoke/20260223_185553/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p6_replace_topq_infer_smoke/20260223_185553/table2_kpis_raw.csv`
  - topq=3（本轮更优候选）
    - run_dir：`runs/v8p6_replace_topq_infer_smoke/20260223_185628`
    - run_json：`runs/v8p6_replace_topq_infer_smoke/20260223_185628/configs/run.json`
    - kpi_mean_raw：`runs/v8p6_replace_topq_infer_smoke/20260223_185628/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p6_replace_topq_infer_smoke/20260223_185628/table2_kpis_raw.csv`
- train+infer smoke：`N/A`
  - train_run：`N/A`
  - infer_run：`N/A`
  - kpi：`N/A`

## 2) 备注

- 本版新增 `--forest-replace-topq`（替换动作候选 Top-Q 约束），回填结果时必须同时记录 `forest_replace_ranking` 与 `forest_replace_topq`。
