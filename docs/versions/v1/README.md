# v1

- 版本类型：**Major (v+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`N/A`

## 概要
- CNN short/long 成功率：`0` / `0`
- Hybrid short/long 成功率：`1` / `1`
- 差值（CNN - Hybrid）：short `-1`, long `-1`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 建立 `strict-argmax`（旧称 strict no-fallback）基线链路，验证可运行与可追溯性。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1.json`、`repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke.json`
- 训练命令：`conda run -n ros2py310 python train.py --profile repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke`
- 推理命令：`conda run -n ros2py310 python infer.py --profile repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke`

## 代表性运行
- run_dir: `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017`
- run_json: `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017/configs/run.json`
- KPI 路径：`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017/infer/20260209_003447/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v2`
- 计划改动：在保持 `strict-argmax`（旧称 strict no-fallback）不变的前提下提高探索效率，避免全 0 成功率。
