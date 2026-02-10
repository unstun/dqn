# v3p8

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）
- 状态：**未通过**
- 上一版本：`v3p7`

## 概要
- 目标：在 short/long 双套件上提升 `strict-argmax`（旧称 strict no-fallback）的 checkpoint 选择稳健性。
- 结果：short/long `成功率=0.4/0.2`，仍低于 Hybrid `0.9/1.0`。
- 决策：保留联合 checkpoint 思路，继续做面向 timeout 的小步调参。

## 方法意图
- 将原先单分布 checkpoint 偏好，替换为 short/long 联合选择。
- 保持算法定义一致：不引入推理期 fallback/replacement/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p8_smoke.json`
- 命令：
  - `conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p8_smoke`
  - `conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p8_smoke`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p8_smoke/train_20260209_182128`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p8_smoke/train_20260209_182128/configs/run.json`
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p8_smoke/train_20260209_182128/infer/20260209_184141/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。
- 主要失败仍为 超时崩塌（尤其 long 套件）。

## 下一步
- 下一版本：`v3p9`
- 计划改动：保留联合 checkpoint 选择，增加轻量 anti-timeout reward/filter 调整。
