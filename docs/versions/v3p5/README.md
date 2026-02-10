# v3p5

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`v3p4`

## 概要
- CNN short/long 成功率：`0.2` / `0`
- Hybrid short/long 成功率：`0.9` / `1`
- 差值（CNN - Hybrid）：short `-0.7`, long `-1`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 回到 v3p1_smoke 主干并固定 final checkpoint，检查稳定性。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`N/A（基于上一版 profile + CLI 覆盖）`
- 训练命令：`conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p5_smoke --episodes 180 --save-ckpt final`
- 推理命令：`N/A（见 run_json 中实际参数）`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p5_smoke/train_20260209_160207`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p5_smoke/train_20260209_160207/configs/run.json`
- KPI 路径：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p5_smoke/train_20260209_160207/infer/20260209_162247/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v3p6`
- 计划改动：比较 `final` 与 `best` checkpoint 的泛化差异。
