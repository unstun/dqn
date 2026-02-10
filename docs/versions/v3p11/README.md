# v3p11

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）
- 状态：**未通过**
- 上一版本：`v3p10`

## 概要
- 目标：在保持 `strict-argmax`（旧称 strict no-fallback）的前提下，恢复 v3p10 的 long 崩塌。
- 结果：short/long `成功率=0.3/0.5`，对照 Hybrid 为 `0.9/1.0`。
- 决策：long 有恢复但 short 回落，继续进行小步平衡调参。

## 方法意图
- 保留双套件训练与联合 checkpoint 选择。
- 仅新增 short-prob ramp 采样调度，不引入任何 fallback 机制。

## 复现实验配置 / 命令
- 配置文件：`configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke.json`
- 命令：
  - `conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke`
  - `conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/configs/run.json`
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/infer/20260209_202232/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。
- 主要失败为 short 套件 timeout 仍偏高。

## 下一步
- 下一版本候选：`v3p12`
- 计划改动：保留 ramp，仅做单变量 short/long 平衡微调。
