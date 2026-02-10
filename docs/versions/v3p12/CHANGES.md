# v3p12 - 变更

## 版本意图
- 在不改变算法命名口径（CNN-DDQN, DQfD）前提下，补齐训练可观测性与止损能力。

## 相对 v3p11 的核心改动
- 推理 strict no-fallback 修正：
  - `forest_vehicle_dqn/cli/infer.py`
  - 当 `--forest-no-fallback` 为真时，推理动作为纯 `argmax(Q)`，不再执行 admissible mask/top-k/replacement/fallback/stop override。
- 训练期新指标（每 10 episodes）：
  - `forest_vehicle_dqn/cli/train.py`
  - 新增固定 short/long 评测集，输出：`sr_short/sr_long/sr_all`、`ratio_short/ratio_long`、`n_valid_short/n_valid_long`、`episode/wall_clock_s`。
- RL 早停（非 pretrain）：
  - warmup 后按 patience 点数监控 `sr_all + distance_ratio` 改善，不改善则触发 `rl_early_stop_plateau`。
- 吞吐熔断：
  - 低 episode 且 wall-clock 超阈值触发 `training_throughput_abnormal`，自动止损。
- 训练元数据落盘：
  - 新增 `configs/train_meta_<env>.json`（记录 stop_reason、episodes_completed、train_progress_rows 等）。
- 训练 CLI 新参数：
  - `--train-eval-*`、`--rl-early-stop-*`、`--throughput-abort-*`。

## 新增/更新配置
- 新增 repro configs：
  - `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a.json`
  - `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b.json`
  - `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c.json`
  - `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a_fast.json`
  - `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b_fast.json`
  - `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast.json`

## 受影响文件
- `forest_vehicle_dqn/cli/train.py`
- `forest_vehicle_dqn/cli/infer.py`
- `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_*.json`
- `runs/v3p12_smoke_full_summary.csv`
- `runs/v3p12_smoke_full_failure_dist.csv`
- `runs/v3p12_train_progress_meta_summary.csv`
- `README.md`
- `README.zh-CN.md`
- `docs/versions/README.md`

## 2026-02-10 追加：`train.py` 训练期实时显示（pygame 小窗口）
- 新增 `forest_vehicle_dqn/live_view_pygame.py`：提供 `TrainLiveViewer`，按 step 刷新 RL 训练轨迹。
- 更新 `forest_vehicle_dqn/cli/train.py`：
  - 新增 CLI：`--live-view`、`--live-view-fps`、`--live-view-window-size`、`--live-view-trail-len`、`--live-view-skip-steps`。
  - 仅在 RL 主训练循环接入 viewer；demo collect / demo pretrain 不显示。
  - 关闭窗口后仅禁用 viewer，不中断训练。
  - `pygame` 缺失时打印安装提示并自动降级无窗口训练。
- 更新 `requirements-optional.txt`：新增 `pygame>=2.5`（可选依赖）。
- 新增复现配置：`configs/repro_20260210_train_live_view_pygame_smoke.json`。
- 更新文档：`README.md`、`README.zh-CN.md`（新增训练实时显示命令与可选依赖安装说明）。



## 2026-02-10 二次验证（已安装 pygame）
- 环境补充：`ros2py310` 已安装 `pygame==2.6.1`（来自 `requirements-optional.txt`）。
- 运行验证：新增一轮 `--live-view` 训练 smoke（`episodes=1`）与一轮 `pygame.QUIT` 自动关窗验证。
- 留档同步：已将命令、run 路径与 `N/A` 项写入 `RESULTS.md` 与 `runs/README.md`。

## 2026-02-10 追加：live-view 碰撞检测框
- 更新 `forest_vehicle_dqn/live_view_pygame.py`：
  - 新增碰撞检测框绘制，优先按 `AMRBicycleEnv` 双圆碰撞几何（`footprint.radius_m + _eps_cell_m`）生成包围框。
  - 无法读取几何参数时自动降级为 agent 网格框，避免崩溃。
  - 碰撞时框颜色切换为告警色。
- 更新 `forest_vehicle_dqn/cli/train.py`：
  - 新增 CLI：`--live-view-collision-box/--no-live-view-collision-box`（默认开启）。
  - `TrainLiveViewer` 构造时传入 `show_collision_box`。
- 更新 `configs/repro_20260210_train_live_view_pygame_smoke.json`：默认启用 `live_view_collision_box=true`。
- 更新 `README.md` 与 `README.zh-CN.md`：补充碰撞检测框开关说明。



## 2026-02-10 修正：碰撞框改为车体外框（OBB）
- 问题：上一版 live-view 碰撞框更接近轴对齐包围框，视觉上与车体朝向不一致。
- 修正：`forest_vehicle_dqn/live_view_pygame.py`
  - 按 `pose_m=(x_m,y_m,psi)` 绘制固定尺寸车体外框（OBB），随航向角 `psi` 旋转。
  - 尺寸由 `TwoCircleFootprint(x1_m,x2_m,radius_m)` 反推车体长宽，并叠加 `_eps_cell_m` 作为碰撞边界余量。
  - 保留回退：若几何参数不可用，自动降级到最小固定框，避免运行中断。
- CLI 保持：`--live-view-collision-box/--no-live-view-collision-box`。
- 影响：仅训练可视化显示，不改变算法与决策路径。
