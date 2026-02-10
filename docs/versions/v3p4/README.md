# v3p4

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`；mask 仅用于统计/诊断）。
- 状态：**未通过**
- 上一版本：`v3p3`

## 概要
- CNN short/long 成功率：`0.1` / `0`
- Hybrid short/long 成功率：`0.9` / `1`
- 差值（CNN - Hybrid）：short `-0.8`, long `-1`
- 主要失败模式仍以 timeout/collision 为主，长时域推进能力不足。

## 方法意图
- 减小动作离散并加强探索，尝试缓解长程推进不足。
- 保持 `strict-argmax` 推理口径一致，不引入推理期 replacement/fallback/takeover 等干预。

## 复现实验配置 / 命令
- 配置文件：`N/A（基于上一版 profile + CLI 覆盖）`
- 训练命令：`conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke --episodes 180 --save-ckpt final --forest-action-delta-dot-bins 7 --forest-action-accel-bins 7 --eps-start 0.30 --eps-decay 8000 --forest-demo-pretrain-steps 20000 --forest-demo-pretrain-early-stop-sr 0.7 --forest-demo-pretrain-early-stop-patience 2`
- 推理命令：`N/A（见 run_json 中实际参数）`

## 代表性运行
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke/train_20260209_154029`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke/train_20260209_154029/configs/run.json`
- KPI 路径：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke/train_20260209_154029/infer/20260209_155954/table2_kpis_mean_raw.csv`

## 结论
- 在正式 `strict-argmax`（旧称 strict no-fallback）对比下仍未超过 Hybrid A*-MPC。

## 下一步
- 下一版本：`v3p5`
- 计划改动：在保持无回退约束下回到稳定主干，减少策略漂移。
