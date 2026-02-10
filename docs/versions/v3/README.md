# v3

- 版本类型：**Major (v+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`v2`

## 概要
- CNN short/long 成功率：`0.5` / `0.1`
- Hybrid short/long 成功率：`0.9` / `1`
- 差值（CNN - Hybrid）：short `-0.4`, long `-0.9`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 围绕动作离散粒度与 admissible 约束做小步调参，改善短程可达性。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3.json`、`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke.json`
- 训练命令：`N/A（该版本留档以推理命令覆盖试验为主）`
- 推理命令：`conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke --models repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200 --forest-action-delta-dot-bins 15 --forest-action-accel-bins 15 --forest-adm-horizon 20 --forest-min-progress-m 0.0 --forest-goal-admissible-relax-factor 2.5 --max-steps 1200`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403/configs/run.json`
- KPI 路径：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v3p1`
- 计划改动：进入 p+1 链路做稳定化，重点压低 timeout 并保持 short 优势。
