# v3p1

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`v3`

## 概要
- CNN short/long 成功率：`0.7` / `0.2`
- Hybrid short/long 成功率：`0.9` / `1`
- 差值（CNN - Hybrid）：short `-0.2`, long `-0.8`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 在 v3 基础上做参数回稳并使用 final checkpoint，对齐正式口径。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1.json`、`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke.json`
- 训练命令：`conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1 --save-ckpt final --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt`
- 推理命令：`conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt/train_20260209_143827`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt/train_20260209_143827/configs/run.json`
- KPI 路径：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt/train_20260209_143827/infer/20260209_145935/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v3p2`
- 计划改动：尝试更快 smoke 试错，定位对 long 套件最敏感参数。
