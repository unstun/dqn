# v8p5 runs 追溯（待回填）

## 1) 代表 run（最小追溯字段）

- 回归（fixed pairs，replace-ranking 消融）：
  - `progress_clearance_q`：
    - run_dir：`runs/v8p5_replace_ranking_regression/20260222_222704`
    - run_json：`runs/v8p5_replace_ranking_regression/20260222_222704/run.json`
    - kpi_mean_raw：`runs/v8p5_replace_ranking_regression/20260222_222704/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p5_replace_ranking_regression/20260222_222704/table2_kpis_raw.csv`
  - `progress_clearance_q` + baseline：
    - run_dir：`runs/v8p5_replace_ranking_regression/20260222_224400`
    - run_json：`runs/v8p5_replace_ranking_regression/20260222_224400/run.json`
    - kpi_mean_raw：`runs/v8p5_replace_ranking_regression/20260222_224400/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p5_replace_ranking_regression/20260222_224400/table2_kpis_raw.csv`
  - `clearance_progress_q`：
    - run_dir：`runs/v8p5_replace_ranking_regression/20260222_223308`
    - run_json：`runs/v8p5_replace_ranking_regression/20260222_223308/run.json`
    - kpi_mean_raw：`runs/v8p5_replace_ranking_regression/20260222_223308/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p5_replace_ranking_regression/20260222_223308/table2_kpis_raw.csv`
  - `q`（基线）：
    - run_dir：`runs/v8p5_replace_ranking_regression/20260222_223339`
    - run_json：`runs/v8p5_replace_ranking_regression/20260222_223339/run.json`
    - kpi_mean_raw：`runs/v8p5_replace_ranking_regression/20260222_223339/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p5_replace_ranking_regression/20260222_223339/table2_kpis_raw.csv`
- smoke（episodes=150, runs=3）：`N/A`
  - train_run：`N/A`
  - infer_run：`N/A`
  - kpi：`N/A`
- infer-only smoke（固定 `v7p1` checkpoint，replace-ranking 消融，runs=3）：
  - `q`：
    - run_dir：`runs/v8p5_replace_ranking_infer_smoke/20260223_172217`
    - run_json：`runs/v8p5_replace_ranking_infer_smoke/20260223_172217/run.json`
    - kpi_mean_raw：`runs/v8p5_replace_ranking_infer_smoke/20260223_172217/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p5_replace_ranking_infer_smoke/20260223_172217/table2_kpis_raw.csv`
  - `progress_clearance_q`：
    - run_dir：`runs/v8p5_replace_ranking_infer_smoke/20260223_172252`
    - run_json：`runs/v8p5_replace_ranking_infer_smoke/20260223_172252/run.json`
    - kpi_mean_raw：`runs/v8p5_replace_ranking_infer_smoke/20260223_172252/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p5_replace_ranking_infer_smoke/20260223_172252/table2_kpis_raw.csv`
  - `clearance_progress_q`：
    - run_dir：`runs/v8p5_replace_ranking_infer_smoke/20260223_172327`
    - run_json：`runs/v8p5_replace_ranking_infer_smoke/20260223_172327/run.json`
    - kpi_mean_raw：`runs/v8p5_replace_ranking_infer_smoke/20260223_172327/table2_kpis_mean_raw.csv`
    - kpi_raw：`runs/v8p5_replace_ranking_infer_smoke/20260223_172327/table2_kpis_raw.csv`

## 2) 备注

- 本版新增 `--forest-replace-ranking`（替换动作排序策略），回填结果时必须同时记录该字段取值（`q` / `progress_clearance_q` / `clearance_progress_q`）。
