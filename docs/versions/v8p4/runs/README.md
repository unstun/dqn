# v8p4 runs 追溯（回归已回填；smoke 待跑）

## 1) 回归（v8p3 smoke failures，infer-only）

- 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p4_smoke_failures_regression`
- 执行位置：`ubuntu-zt`（远端优先）
- pairs：`configs/pairs_v8p3_smoke_failures.json`
- `run_dir`: `runs/v8p4_smoke_failures_regression/20260223_142739`
- `run_json`: `runs/v8p4_smoke_failures_regression/20260223_142739/configs/run.json`
- `kpi_mean_raw`: `runs/v8p4_smoke_failures_regression/20260223_142739/table2_kpis_mean_raw.csv`
- `kpi_raw`: `runs/v8p4_smoke_failures_regression/20260223_142739/table2_kpis_raw.csv`

## 2) smoke（episodes=150, runs=3）

### train
- 命令：`conda run -n ros2py310 python train.py --profile repro_20260223_v8p4_fallback_h1_smoke`
- `run_dir`: `N/A`
- `run_json`: `N/A`

### infer
- 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p4_fallback_h1_smoke`
- `run_dir`: `N/A`
- `run_json`: `N/A`
- `kpi_mean_raw`: `N/A`
- `kpi_raw`: `N/A`
