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
- train+infer smoke（topq=3；episodes=140/150 early-stop；runs=3；NO-GO：short/mid collision=1/3）
  - 命令：
    - `conda run -n ros2py310 python train.py --profile v8p6 --forest-replace-topq 3`
    - `conda run -n ros2py310 python infer.py --profile v8p6 --forest-replace-topq 3`
  - train_run：`runs/v8p6_replace_topq_smoke/train_20260223_191450`
  - infer_run：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_192545`
  - run_json（train）：`runs/v8p6_replace_topq_smoke/train_20260223_191450/configs/run.json`
  - run_json（infer）：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_192545/configs/run.json`
  - kpi_mean_raw：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_192545/table2_kpis_mean_raw.csv`
  - kpi_raw：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_192545/table2_kpis_raw.csv`

## 2) 备注

- 本版新增 `--forest-replace-topq`（替换动作候选 Top-Q 约束），回填结果时必须同时记录 `forest_replace_ranking` 与 `forest_replace_topq`。
