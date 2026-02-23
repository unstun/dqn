# v8p2 runs 追溯（已回填）

## 1) infer-only smoke（固定 v7p1 checkpoint，runs=3）

固定模型（checkpoint）：`runs/v7p1_train300_esbest/train_20260221_010743`

- D0：`dijkstra8_nocorner`（`w_clearance=2.0`，`sigma_m=0.5`）：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke`
  - `run_dir`: `runs/v8p2_costmap_infer_smoke/20260223_104100`
  - `run_json`: `runs/v8p2_costmap_infer_smoke/20260223_104100/configs/run.json`
  - `kpi_mean_raw`: `runs/v8p2_costmap_infer_smoke/20260223_104100/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p2_costmap_infer_smoke/20260223_104100/table2_kpis_raw.csv`
- E0：`euclid`（对照）：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke --forest-progress-dist-mode euclid`
  - `run_dir`: `runs/v8p2_costmap_infer_smoke/20260223_104135`
  - `run_json`: `runs/v8p2_costmap_infer_smoke/20260223_104135/configs/run.json`
  - `kpi_mean_raw`: `runs/v8p2_costmap_infer_smoke/20260223_104135/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p2_costmap_infer_smoke/20260223_104135/table2_kpis_raw.csv`
- D1：`dijkstra8_nocorner`（消融：`w_clearance=0.0`，`sigma_m=0.5`）：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke --forest-progress-cost-w-clearance 0`
  - `run_dir`: `runs/v8p2_costmap_infer_smoke/20260223_104209`
  - `run_json`: `runs/v8p2_costmap_infer_smoke/20260223_104209/configs/run.json`
  - `kpi_mean_raw`: `runs/v8p2_costmap_infer_smoke/20260223_104209/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p2_costmap_infer_smoke/20260223_104209/table2_kpis_raw.csv`

## 2) train+infer smoke（episodes=150, runs=3）
- train：
  - 命令：`conda run -n ros2py310 python train.py --profile repro_20260223_v8p2_costmap_smoke`
  - `run_dir`: `runs/v8p2_costmap_smoke/train_20260223_104408`
  - `run_json`: `runs/v8p2_costmap_smoke/train_20260223_104408/configs/run.json`
  - `train_meta`: `runs/v8p2_costmap_smoke/train_20260223_104408/configs/train_meta_forest_a.json`
  - `train_flow`: `runs/v8p2_costmap_smoke/train_20260223_104408/train_flow.log`
- infer（I0，seed=33）：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_smoke`
  - `run_dir`: `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027`
  - `run_json`: `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027/configs/run.json`
  - `kpi_mean_raw`: `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027/table2_kpis_raw.csv`
- infer（I1，seed=34 复测）：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_smoke --seed 34`
  - `run_dir`: `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110608`
  - `run_json`: `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110608/configs/run.json`
  - `kpi_mean_raw`: `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110608/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110608/table2_kpis_raw.csv`
