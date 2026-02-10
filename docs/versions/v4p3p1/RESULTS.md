# v4p3p1 结果

## 代表性正式指标（当前）
- KPI 路径：`runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044/table2_kpis_mean_raw.csv`
- CNN short/long 成功率：`0.0` / `0.0`
- Hybrid short/long 成功率：`0.9` / `1.0`
- CNN argmax 不可行动作率 short/long：`0.438` / `0.355`

## 与上一版对比（v4p3）
- `v4p3`：short/long=`0.2/0.0`
- `v4p3p1`：short/long=`0.0/0.0`
- 结论：本轮单变量改动导致 short 退化，long 无改善。

## 门槛检查（对标 Hybrid A*-MPC）
- `success_rate(CNN) >= success_rate(Hybrid)`: short `未通过`（0.0 < 0.9）, long `未通过`（0.0 < 1.0）
- `avg_path_length(CNN) < avg_path_length(Hybrid)`: short `N/A`, long `N/A`
- `path_time_s(CNN) < path_time_s(Hybrid)`: short `N/A`, long `N/A`
- 最终门槛状态：`未通过`

## 运行过程记录（2026-02-10）

### self-check
- command:
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（`device_ok=cuda:0`）

### train（300 episodes）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958/configs/run.json`
- train_eval: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958/training_eval.csv`
- 关键训练日志：
  - `Train-progress ep=300: sr(short/long/all)=0.600/0.000/0.300`
  - `save_ckpt joint short/long: chosen=final, short=0.600, long=0.600`

### infer（runs=10）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958/models --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.0/0.0`；`argmax_inad(short/long)=0.438/0.355`。
- failure_reason（raw）：short `collision=10`；long `collision=10`。

## 结论
- `v4p3p1` 未达到预期，单变量调整 short 采样比例不能解决 long 泛化问题。
- 下一轮应改课程动态参数或 replay 机制，而不是继续只调 short_prob。
