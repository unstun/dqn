# v3p2

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`v3p1`

## 概要
- CNN short/long 成功率：`0.2` / `0.1`
- Hybrid short/long 成功率：`0.9` / `1`
- 差值（CNN - Hybrid）：short `-0.7`, long `-0.9`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 采用更短 smoke 回路（`episodes=1`）做快速方向性筛查。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2.json`、`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke.json`
- 训练命令：`conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke --episodes 1 --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke_ep1`
- 推理命令：`conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke_ep1/train_20260209_133603`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke_ep1/train_20260209_133603/configs/run.json`
- KPI 路径：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke_ep1/train_20260209_133603/infer/20260209_135155/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v3p3`
- 计划改动：恢复常规 smoke 并验证可重复性，避免一次性偶然结果。
