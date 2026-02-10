# v3p9

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）
- 状态：**未通过**
- 上一版本：`v3p8`

## 概要
- 目标：在保留 v3p8 联合 checkpoint 选择的前提下，降低超时崩塌。
- 结果：short/long `成功率=0.2/0.2`，仍低于 Hybrid `0.9/1.0`。
- 决策：该调参导致 short 回退且 long 未改善，继续转向采样分布方向。

## 方法意图
- 保持 `strict-argmax`（旧称 strict no-fallback）与联合 checkpoint 选择。
- 仅做轻量 anti-timeout reward/filter 参数调整。

## 复现实验配置 / 命令
- 配置文件：`configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p9_smoke.json`
- 命令：
  - `conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p9_smoke`
  - `conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p9_smoke`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p9_smoke/train_20260209_184524`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p9_smoke/train_20260209_184524/configs/run.json`
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p9_smoke/train_20260209_184524/infer/20260209_190223/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。
- timeout 仍是主导失败模式。

## 下一步
- 下一版本：`v3p10`
- 计划改动：将训练随机对采样切换为 short/long 双套件分布对齐。
