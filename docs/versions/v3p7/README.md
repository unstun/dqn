# v3p7

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`v3p6`

## 概要
- CNN short/long 成功率：`0` / `0`
- Hybrid short/long 成功率：`0.9` / `1`
- 差值（CNN - Hybrid）：short `-0.9`, long `-1`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 切换 `legacy` demo 模式并启用分层 replay，观察失败结构变化。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`N/A（基于上一版 profile + CLI 覆盖）`
- 训练命令：`conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p7_legacy_smoke --demo-mode legacy --episodes 140 --save-ckpt final --replay-stratified --replay-frac-demo 0.5 --replay-frac-goal 0.2 --replay-frac-stuck 0.15 --replay-frac-hazard 0.15 --demo-lambda 4.0 --demo-margin 1.0`
- 推理命令：`N/A（见 run_json 中实际参数）`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p7_legacy_smoke/train_20260209_170153`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p7_legacy_smoke/train_20260209_170153/configs/run.json`
- KPI 路径：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p7_legacy_smoke/train_20260209_170153/infer/20260209_172419/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v3p8`
- 计划改动：进入联合 short/long checkpoint 选择（v3p8）。
