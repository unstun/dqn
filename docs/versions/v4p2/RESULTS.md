# v4p2 结果

## 代表性正式指标（当前 best）
- KPI 路径：`runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730/table2_kpis_mean_raw.csv`
- CNN short/long 成功率：`0.0` / `0.0`
- Hybrid short/long 成功率：`0.9` / `1.0`
- CNN argmax 不可行动作率 short/long：`0.478` / `0.477`

## 三轮 smoke 对比（runs=10）
| 轮次 | aux λ | short SR | long SR | argmax inad (short/long) | failure_reason(short) | failure_reason(long) |
|---|---:|---:|---:|---|---|---|
| iter1 | 0.20 | 0.0 | 0.0 | 0.478 / 0.477 | collision=10 | collision=10 |
| iter2 | 0.05 | 0.0 | 0.0 | 0.509 / 0.449 | collision=10 | collision=10 |
| iter3 | 0.01 | 0.0 | 0.0 | 0.560 / 0.351 | collision=9, timeout=1 | collision=7, timeout=3 |

## 门槛检查（对标 Hybrid A*-MPC）
- `success_rate(CNN) >= success_rate(Hybrid)`: short `未通过`（0.0 < 0.9）, long `未通过`（0.0 < 1.0）
- `avg_path_length(CNN) < avg_path_length(Hybrid)`: short `N/A`（CNN short 无成功轨迹）, long `N/A`
- `path_time_s(CNN) < path_time_s(Hybrid)`: short `N/A`, long `N/A`
- 最终门槛状态：`未通过`

## 运行过程记录（2026-02-10）

### 自检
- command:
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（`device_ok=cuda:0`）

### quick smoke（iter1，aux=0.2）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 4000 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02/train_20260210_145239/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10`
- run_dir(train): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02/train_20260210_145239`
- run_json(train): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02/train_20260210_145239/configs/run.json`
- run_dir(infer): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730`
- run_json(infer): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.0/0.0`；`argmax_inad(short/long)=0.478/0.477`。
- failure_reason（raw）：short `collision=10`；long `collision=10`。

### quick smoke（iter2，aux=0.05）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 4000 --aux-admissibility-lambda 0.05 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005/train_20260210_145857/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005_infer10`
- run_dir(train): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005/train_20260210_145857`
- run_json(train): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005/train_20260210_145857/configs/run.json`
- run_dir(infer): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005_infer10/20260210_150353`
- run_json(infer): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005_infer10/20260210_150353/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005_infer10/20260210_150353/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.0/0.0`；`argmax_inad(short/long)=0.509/0.449`。
- failure_reason（raw）：short `collision=10`；long `collision=10`。

### quick smoke（iter3，aux=0.01）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 4000 --aux-admissibility-lambda 0.01 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001/train_20260210_151314/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001_infer10`
- run_dir(train): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001/train_20260210_151314`
- run_json(train): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001/train_20260210_151314/configs/run.json`
- run_dir(infer): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001_infer10/20260210_151812`
- run_json(infer): `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001_infer10/20260210_151812/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001_infer10/20260210_151812/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.0/0.0`；`argmax_inad(short/long)=0.560/0.351`。
- failure_reason（raw）：short `collision=9, timeout=1`；long `collision=7, timeout=3`。

## 结论
- `v4p2` 的 aux 训练约束已完成三轮单变量扫参，但成功率仍停留在 `0.0/0.0`。
- 在当前 smoke 预算下，aux λ 的变化主要影响 `argmax_inad` 分布，尚未转化为成功率收益。
- 建议结束 `v4p2` 的 aux-only 验证，转入下一增量版本（如 `v4p3`）测试 replay/课程机制。
