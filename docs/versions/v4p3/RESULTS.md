# v4p3 结果

## 代表性正式指标（当前）
- KPI 路径：`runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934/table2_kpis_mean_raw.csv`
- CNN short/long 成功率：`0.2` / `0.0`
- Hybrid short/long 成功率：`0.9` / `1.0`
- CNN argmax 不可行动作率 short/long：`0.460` / `0.509`

## 门槛检查（对标 Hybrid A*-MPC）
- `success_rate(CNN) >= success_rate(Hybrid)`: short `未通过`（0.2 < 0.9）, long `未通过`（0.0 < 1.0）
- `avg_path_length(CNN) < avg_path_length(Hybrid)`: short `未通过`（19.6259 > 15.5051）, long `N/A`（long 无成功轨迹）
- `path_time_s(CNN) < path_time_s(Hybrid)`: short `未通过`（14.35 > 9.0556）, long `N/A`
- 最终门槛状态：`未通过`

## 运行过程记录（2026-02-10）

### self-check
- command:
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（`device_ok=cuda:0`）

### train（300 episodes）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001/train_20260210_152732`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001/train_20260210_152732/configs/run.json`
- train_eval: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001/train_20260210_152732/training_eval.csv`
- 关键训练日志：
  - `Train-progress ep=300: sr(short/long/all)=0.600/0.000/0.300`
  - `save_ckpt joint short/long: chosen=final, short=0.600, long=0.500`

### infer（runs=10）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300 --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001/train_20260210_152732/models --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.2/0.0`；`argmax_inad(short/long)=0.460/0.509`。
- failure_reason（raw）：short `collision=8, reached=2`；long `collision=10`。

## 结论
- `v4p3` 证明“300 轮训练”能带来 short 恢复（从 `0.0` 提升到 `0.2`）。
- 但 long 套件仍为 `0.0`，说明当前课程/采样设置仍偏短程，长程泛化不足。
- 下一轮应继续单变量迭代，目标是先把 long 拉出 0 再谈 full20。
