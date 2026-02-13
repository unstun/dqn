# v6p2 - 运行记录

> 说明：本版不新增训练/推理实验；仍记录本轮执行的 self-check，并引用 v6p1 的代表性 KPI 路径作为“结果继承”的可追溯来源。

## 自检（本轮新增）
- `bash scripts/self_check.sh`（通过）
- `conda run -n ros2py310 python scripts/repo_doctor.py --strict`（通过）
- `conda run -n ros2py310 python game.py --self-check`（通过）

## 交互式 demo
- `conda run -n ros2py310 python game.py --profile repro_20260212_interactive_game_forest_a_v1`（交互式；无 run_dir）

## 继承 v6p1 的代表性正式 runs（fixed pairs20）
- long（v6p1 long KPI）：
  - run_dir：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414`
  - run_json：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414/table2_kpis_mean_raw.csv`
- short（v6p1 short KPI）：
  - run_dir：`runs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1/20260212_003744`
  - run_json：`runs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1/20260212_003744/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1/20260212_003744/table2_kpis_mean_raw.csv`

