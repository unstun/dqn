# v2

- 版本类型：**Major (v+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`v1`

## 概要
- CNN short/long 成功率：`0` / `0`
- Hybrid short/long 成功率：`1` / `1`
- 差值（CNN - Hybrid）：short `-1`, long `-1`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 提升探索并缩短训练周期，观察是否可快速拉起成功率。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2.json`、`repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke.json`
- 训练命令：`conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke`
- 推理命令：`conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246/configs/run.json`
- KPI 路径：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246/infer/20260209_083902/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v3`
- 计划改动：从动作/约束口径继续小步调参，优先提升 short 套件成功率。
