# v8p1 runs 追溯（待回填）

## 1) infer-only smoke（固定 v7p1 checkpoint，对照）
- `grid4`：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_infer_smoke`
  - `run_dir`: `N/A`
  - `run_json`: `N/A`
  - `kpi_mean_raw`: `N/A`
  - `kpi_raw`: `N/A`
- `euclid`（对照）：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_infer_smoke --forest-progress-dist-mode euclid`
  - `run_dir`: `N/A`
  - `run_json`: `N/A`
  - `kpi_mean_raw`: `N/A`
  - `kpi_raw`: `N/A`

## 2) train+infer smoke（episodes=150, runs=3）
- train：
  - 命令：`conda run -n ros2py310 python train.py --profile repro_20260222_v8p1_navdist_smoke`
  - `run_dir`: `N/A`
  - `run_json`: `N/A`
  - `train_meta`: `N/A`
  - `train_flow`: `N/A`
- infer：
  - 命令：`conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_smoke`
  - `run_dir`: `N/A`
  - `run_json`: `N/A`
  - `kpi_mean_raw`: `N/A`
  - `kpi_raw`: `N/A`

