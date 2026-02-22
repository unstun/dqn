# v8p1 runs 追溯（已回填）

## 1) infer-only smoke（固定 v7p1 checkpoint，对照）
- `grid4`：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_infer_smoke`
  - `run_dir`: `runs/v8p1_navdist_infer_smoke/20260223_021151`
  - `run_json`: `runs/v8p1_navdist_infer_smoke/20260223_021151/configs/run.json`
  - `kpi_mean_raw`: `runs/v8p1_navdist_infer_smoke/20260223_021151/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p1_navdist_infer_smoke/20260223_021151/table2_kpis_raw.csv`
- `euclid`（对照）：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_infer_smoke --forest-progress-dist-mode euclid`
  - `run_dir`: `runs/v8p1_navdist_infer_smoke/20260223_021220`
  - `run_json`: `runs/v8p1_navdist_infer_smoke/20260223_021220/configs/run.json`
  - `kpi_mean_raw`: `runs/v8p1_navdist_infer_smoke/20260223_021220/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p1_navdist_infer_smoke/20260223_021220/table2_kpis_raw.csv`

## 2) train+infer smoke（episodes=150, runs=3）
- train：
  - 命令：`conda run -n ros2py310 python train.py --profile repro_20260222_v8p1_navdist_smoke`
  - `run_dir`: `runs/v8p1_navdist_smoke/train_20260223_021339`
  - `run_json`: `runs/v8p1_navdist_smoke/train_20260223_021339/configs/run.json`
  - `train_meta`: `runs/v8p1_navdist_smoke/train_20260223_021339/configs/train_meta_forest_a.json`
  - `train_flow`: `runs/v8p1_navdist_smoke/train_20260223_021339/train_flow.log`
- infer：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_smoke --models runs/v8p1_navdist_smoke/train_20260223_021339`
  - `run_dir`: `runs/v8p1_navdist_smoke/train_20260223_021339/infer/20260223_023932`
  - `run_json`: `runs/v8p1_navdist_smoke/train_20260223_021339/infer/20260223_023932/configs/run.json`
  - `kpi_mean_raw`: `runs/v8p1_navdist_smoke/train_20260223_021339/infer/20260223_023932/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p1_navdist_smoke/train_20260223_021339/infer/20260223_023932/table2_kpis_raw.csv`
