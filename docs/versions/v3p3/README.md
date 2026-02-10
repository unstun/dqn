# v3p3

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`v3p2`

## 概要
- CNN short/long 成功率：`0.2` / `0.2`
- Hybrid short/long 成功率：`0.9` / `1`
- 差值（CNN - Hybrid）：short `-0.7`, long `-0.8`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 恢复标准 smoke 训练回路，验证 v3p2 方向可迁移性。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3.json`、`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke.json`
- 训练命令：`conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke`
- 推理命令：`conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke/train_20260209_135600`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke/train_20260209_135600/configs/run.json`
- KPI 路径：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke/train_20260209_135600/infer/20260209_141406/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v3p4`
- 计划改动：继续以 timeout 为主线调参，逐步增加 long 可达概率。
