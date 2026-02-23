# v8p3 runs 追溯（待回填）

## 1) 回归（固定 v8p2 short collision pair，infer-only）
- 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p3_short_collision_pair_regression`
- `run_dir`: `runs/v8p3_short_collision_pair_regression/20260223_124513`
- `run_json`: `runs/v8p3_short_collision_pair_regression/20260223_124513/configs/run.json`
- `kpi_mean_raw`: `runs/v8p3_short_collision_pair_regression/20260223_124513/table2_kpis_mean_raw.csv`
- `kpi_raw`: `runs/v8p3_short_collision_pair_regression/20260223_124513/table2_kpis_raw.csv`

### strict-argmax（诊断）
- 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p3_short_collision_pair_regression --forest-no-fallback --out v8p3_short_collision_pair_regression_strict`
- `run_dir`: `runs/v8p3_short_collision_pair_regression_strict/20260223_124959`
- `run_json`: `runs/v8p3_short_collision_pair_regression_strict/20260223_124959/configs/run.json`
- `kpi_mean_raw`: `runs/v8p3_short_collision_pair_regression_strict/20260223_124959/table2_kpis_mean_raw.csv`
- `kpi_raw`: `runs/v8p3_short_collision_pair_regression_strict/20260223_124959/table2_kpis_raw.csv`

## 2) smoke（episodes=150, runs=3）

### train
- 命令：`conda run -n ros2py310 python train.py --profile repro_20260223_v8p3_fallback_safety_smoke`
- `run_dir`: `runs/v8p3_fallback_safety_smoke/train_20260223_125609`
- `run_json`: `runs/v8p3_fallback_safety_smoke/train_20260223_125609/configs/run.json`

### infer
- 命令：`conda run -n ros2py310 python infer.py --profile repro_20260223_v8p3_fallback_safety_smoke`
- `run_dir`: `runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153`
- `run_json`: `runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153/configs/run.json`
- `kpi_mean_raw`: `runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153/table2_kpis_mean_raw.csv`
- `kpi_raw`: `runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153/table2_kpis_raw.csv`
