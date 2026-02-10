# v4p1

- 版本类型：**Major 线 v4 的首个补丁版（p+1）**
- 研究路线：`CNN-DDQN v4` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`）
- 状态：**未通过（quick smoke 已完成）**
- 上一版本：`v3p12`

## 概要
- 目标：在保持推理期纯 `argmax(Q)` 的前提下，将 HOPE 可迁移思想落到训练期（动态课程 + 失败感知回放），并加入 long 稳定性约束。
- 本版聚焦：先做 `v4p1` 最小高置信实现并跑通 smoke，确认是否比 `v3p12`/`v3p12recover` 有恢复趋势。
- 当前进度：代码与配置已落地，`self-check` 已通过，已完成两轮 quick smoke（`runs=10` 评测）。

## 方法意图
- **动态 short/long 课程（训练期）**：根据 train-progress 的 short/long SR 偏差动态调整 `p_short`。
- **failure-aware PER（训练期）**：对 `near_goal/stuck/hazard` 标记样本施加 PER priority boost，优先学习失败前驱。
- **checkpoint long 下限（训练期）**：联合 short/long 选 ckpt 时加入 `long SR floor`，抑制 short 偏科。
- **`strict-argmax`（推理期）**：保持纯 `argmax(Q)`，不引入 masking/top-k/replacement/fallback/接管。

## 复现实验配置 / 命令
- 主配置：`configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke.json`

- 自检：
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`

- quick smoke（已执行，推荐复现命令）：
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 4000 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k/train_20260210_135138/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10`

## 代表性运行
- 训练 run：`runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k/train_20260210_135138`
- run_json：`runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k/train_20260210_135138/configs/run.json`
- KPI：`runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10/20260210_135524/table2_kpis_mean_raw.csv`

## 结论
- 当前 best（quick smoke）为 `CNN short/long = 0.1 / 0.0`，低于 `Hybrid 0.9 / 1.0`，尚未通过。
- 但较无 demo 的 quick smoke（`0.0 / 0.0`）有轻微恢复（short +0.1），说明 failure-aware + 课程方向仍可继续挖。

## 下一步
- 进入 `v4p2`：优先降低 `argmax_inadmissible_rate`（当前 short/long 约 `0.483/0.490`）。
- 先做单变量：动作空间离散度、`forest_min_progress_m`、`forest_adm_horizon` 与 `dynamic_k` 的组合扫描。
- smoke 达到 `short>=0.3,long>=0.3` 后，再进 full `runs=20`。
