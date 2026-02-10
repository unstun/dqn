# v3p10

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）
- 状态：**未通过**
- 上一版本：`v3p9`

## 概要
- 目标：让训练随机对分布与推理 short/long 双套件分布对齐。
- 结果：short/long `成功率=0.4/0.0`；long 出现完全 timeout 崩塌。
- 决策：保留双套件方向，但加入 short-prob ramp 稳定训练进程。

## 方法意图
- 在不引入任何 fallback 机制的前提下，加入训练期双套件采样。
- 保持 `strict-argmax`（旧称 strict no-fallback）与联合 checkpoint 选择不变。

## 复现实验配置 / 命令
- 配置文件：`configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p10_smoke.json`
- 命令：
  - `conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p10_smoke`
  - `conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p10_smoke`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p10_smoke/train_20260209_194139`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p10_smoke/train_20260209_194139/configs/run.json`
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p10_smoke/train_20260209_194139/infer/20260209_195717/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。
- long 套件 超时崩塌 恶化为 `10/10` timeout。

## 下一步
- 下一版本：`v3p11`
- 计划改动：保留双套件并加入 short-prob ramp（前期 short 偏重，后期增强 long 暴露）。
