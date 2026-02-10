# v3p12

- 版本类型：**Minor (p+1)**
- 研究路线：`CNN-DDQN` 推理口径：`strict-argmax`（旧称 strict no-fallback；推理期纯 `argmax(Q)`）
- 状态：**未通过**
- 上一版本：`v3p11`

## 概要
- 目标：在 `strict-argmax`（旧称 strict no-fallback）硬约束下，补齐训练期关键指标（SR 曲线、distance_ratio 曲线）、RL早停与异常耗时熔断，并完成 smoke→full 门控评测。
- 结果：`smoke(a/b/c)` 与 `full(runs=20)` 均未恢复成功率；`full` 中 CNN short/long 成功率为 `0.0/0.0`，Hybrid 为 `0.95/1.0`。
- 结论：本版主要完成“可观测性+止损机制”建设，性能门槛未达成。

## 方法意图
- `strict-argmax` 合规：推理期纯 `argmax(Q)`，禁用 masking/top-k/stop-override/replacement/fallback/heuristic takeover。
- 新增训练期固定评测集：每 10 episodes 评测 `sr_short/sr_long/sr_all` 与 `ratio_short/ratio_long`。
- 新增 RL 早停：warmup 后以 `sr_all + distance_ratio` 联合判据检测 plateau。
- 新增吞吐熔断：低 episode 且超时异常时中止 run，记录 `training_throughput_abnormal`。

## 复现实验配置 / 命令
- 主 smoke profile：`configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast.json`
- full 评测命令（runs=20）：
  - `conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast --models runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast/train_20260210_021930/models --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast --runs 20`

## 代表性运行
- smoke 主 run：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast/train_20260210_021930`
- full run：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622`
- 指标汇总：`runs/v3p12_smoke_full_summary.csv`

## 结论
- P0（学术定义+`strict-argmax`）：**实现层面满足**（推理 `fallback_rate` 为 0.0）。
- P1（RL 100% SR short/long）：**未达到**。
- P2（同时超过 Hybrid A*-MPC 三条件）：**未达到**。
- P3（减少无效训练、异常止损）：**达到（机制已生效）**。
- 失败运行已归档：含 2 个人工中止 run（`N/A` + failure_reason）。

## 下一步
- 继续在同一评测框架下定位“高 argmax 不可行动作率→碰撞主导”的根因。
- 下一版优先针对动作空间/奖励权重做小步单变量试验，并保持 `strict-argmax`（旧称 strict no-fallback）不变。

## 2026-02-10 补充（训练实时显示）
- 目的：为 `train.py` 增加 RL 阶段实时可视化（pygame 小窗口），用于训练过程观察。
- 配置：`configs/repro_20260210_train_live_view_pygame_smoke.json`
- headless 验证 run：`runs/repro_20260210_train_live_view_pygame_smoke_headless/train_20260210_113131`
- 结果：`pygame` 缺失场景下可自动降级无窗口并继续训练（`stop_reason=completed`）。
- 说明：Windows 桌面窗口人工观测仍待补充，当前记录为 `N/A`。



## 2026-02-10 二次验证（已安装 pygame）
- 新增验证 run：`runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849`（`--live-view` 开启）。
- 行为验证：`pygame.QUIT` 注入后 `TrainLiveViewer` 输出“window closed by user; training continues”并进入 disabled。
- 当前结论：训练期实时显示链路已可用；Windows 人工观测窗口体验可按 README 命令继续做本机验证。

## 2026-02-10 补充（碰撞检测框）
- 训练实时窗口新增碰撞检测框（默认开启，支持 `--no-live-view-collision-box`）。
- 关键验证 run：
  - 开启：`runs/repro_20260210_train_live_view_pygame_box_on/train_20260210_115024`
  - 关闭：`runs/repro_20260210_train_live_view_pygame_box_off/train_20260210_115102`
- 两轮均 `stop_reason=completed`，说明该功能未破坏训练流程。



## 2026-02-10 补充（OBB 碰撞框）
- 碰撞框口径修正为“固定车体外框（OBB）+ 航向角旋转”，不再使用轴对齐近似框。
- 关键验证 run：
  - 开启：`runs/repro_20260210_train_live_view_pygame_obb_on/train_20260210_122251`
  - 关闭：`runs/repro_20260210_train_live_view_pygame_obb_off/train_20260210_122326`
- 两轮均 `stop_reason=completed`，功能变更未影响训练主流程。
