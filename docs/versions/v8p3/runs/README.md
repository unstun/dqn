# v8p3 runs 追溯（待回填）

## 1) 回归（固定 v8p2 short collision pair，infer-only）
- 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p3_short_collision_pair_regression`
- `run_dir`: N/A
- `run_json`: N/A
- `kpi_mean_raw`: N/A
- `kpi_raw`: N/A

## 2) smoke（episodes=150, runs=3）

### train
- 命令：`conda run -n ros2py310 python train.py --profile repro_20260223_v8p3_fallback_safety_smoke`
- `run_dir`: N/A
- `run_json`: N/A

### infer
- 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p3_fallback_safety_smoke`
- `run_dir`: N/A
- `run_json`: N/A
- `kpi_mean_raw`: N/A
- `kpi_raw`: N/A

