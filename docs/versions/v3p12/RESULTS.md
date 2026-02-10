# v3p12 - 结果

## 数据来源
- smoke/full 汇总：`runs/v3p12_smoke_full_summary.csv`
- failure 分布汇总：`runs/v3p12_smoke_full_failure_dist.csv`
- train-progress 汇总：`runs/v3p12_train_progress_meta_summary.csv`
- full runs=20 KPI：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622/table2_kpis_mean_raw.csv`

## smoke 结果（forest_a）
- 候选：`a_fast / b_fast / c_fast`
- 共同现象：CNN short/long 成功率均为 `0.0/0.0`；Hybrid 约 `0.833/1.0`。
- strict no-fallback 证据：CNN `fallback_rate=0.0`（各候选一致）。

## full 结果（short/long, runs=20）
- run: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622`
- CNN success_rate：short `0.0`，long `0.0`
- Hybrid success_rate：short `0.95`，long `1.0`
- CNN `fallback_rate`：short `0.0`，long `0.0`
- CNN `argmax_inadmissible_rate`：short `0.457`，long `0.455`

## 门槛检查（对 Hybrid A*-MPC）
- `success_rate(CNN) >= success_rate(Hybrid)`：short `FAIL`，long `FAIL`
- `avg_path_length(CNN) < avg_path_length(Hybrid)`：short `FAIL`（CNN 无成功样本，N/A），long `FAIL`（N/A）
- `path_time_s(CNN) < path_time_s(Hybrid)`：short `FAIL`（N/A），long `FAIL`（N/A）
- 最终门槛状态：`未通过`

## failure_reason 分布
- full short：`collision:18, timeout:2`
- full long：`collision:20`
- 主要瓶颈：高碰撞占比 + 高 argmax 不可行动作率（但保持 strict no-fallback 无接管）。

## 失败运行（N/A 归档）
- 手工终止 smoke-b 原始长跑：
  - run: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b/train_20260210_023020`
  - 状态：`N/A`（人工中止，避免继续占用预算）
  - 原因：`manual_stop_time_budget`（该轮未产出可用 KPI 文件）
- 手工终止 smoke-a 原始长跑：
  - run: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a/train_20260210_014204`
  - 状态：`N/A`（人工中止，转入更快 smoke 配置）
  - 原因：`manual_stop_time_budget`（该轮未产出可用 KPI 文件）

## 训练期指标与早停/熔断
- train-progress 指标文件：各 run 的 `training_eval.csv`（含 `phase=train` 行）。
- smoke fast 三候选均产出 `train_progress_rows=9`，`stop_reason=completed`。
- 熔断机制验证 run：
  - `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_throughput_test/train_20260210_015016`
  - `stop_reason=training_throughput_abnormal`，`episodes_completed=1/3`（按阈值主动止损）。

## 2026-02-10 训练实时显示验证（RL 阶段）
- 目标：验证 `train.py` 训练期 live-view 的可用性与容错，不改变算法定义。

### 自检
- 命令：
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（CUDA 可用，`device_ok=cuda:0`）。

### smoke（headless + live-view 容错）
- 命令：
  - `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_smoke_headless`
- run_dir：`runs/repro_20260210_train_live_view_pygame_smoke_headless/train_20260210_113131`
- 关键产物：
  - `configs/run.json`
  - `configs/train_meta_forest_a.json`
  - `training_returns.csv`
- 训练状态：`episodes_completed=1/1`，`stop_reason=completed`。
- live-view 容错：检测到 `pygame` 缺失后打印安装提示并继续训练，最终正常结束（`PASS`）。



### 默认无影响验证（`--no-live-view`）
- 命令：
  - `conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --no-live-view --out repro_20260210_train_live_view_pygame_smoke_noview`
- run_dir：`runs/repro_20260210_train_live_view_pygame_smoke_noview/train_20260210_113351`
- 训练状态：`episodes_completed=1/1`，`stop_reason=completed`（默认关闭/显式关闭均不影响训练流程）。

### 失败运行（N/A）
- 命令：
  - `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 120 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_smoke_headless`
- 状态：`N/A`（启动前参数校验失败）
- 原因：`max_steps_too_small_for_forest_env`（报错要求至少 `511` steps）

### GUI 手工验证（Windows）
- 状态：`N/A`（当前轮次在 headless 环境执行，未进行桌面窗口人工观测）。
- 建议命令：
  - `conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --live-view --live-view-fps 0 --live-view-window-size 900 --live-view-trail-len 300 --live-view-skip-steps 1`

### short/long KPI 与 failure_reason
- 本轮为训练可视化 smoke 验证，未执行 infer 评测：
  - `short/long KPI`: `N/A`（无 `table2_kpis_mean_raw.csv`）
  - `failure_reason` 分布：`N/A`（未产出 infer 原始 KPI）



## 2026-02-10 二次验证（已安装 pygame）

### 训练 smoke（`--live-view` 开启）
- 命令：
  - `conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_gui_try`
- run_dir：`runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849`
- 关键产物：
  - `runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849/configs/run.json`
  - `runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849/configs/train_meta_forest_a.json`
  - `runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849/training_returns.csv`
- 结果：`episodes_completed=1/1`，`stop_reason=completed`。
- 备注：日志未出现 live-view 初始化失败/降级提示；本轮在 CLI 环境无法人工肉眼确认窗口内容（记为 `N/A`）。

### 关窗不中断自动化验证（`pygame.QUIT` 注入）
- 命令：
  - `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python -c "import sys;from pathlib import Path;sys.path.insert(0,str(Path('/home/sun/phdproject/dqn/dqn')));import pygame;from forest_vehicle_dqn.live_view_pygame import TrainLiveViewer;from forest_vehicle_dqn.maps import get_map_spec;from forest_vehicle_dqn.env import AMRBicycleEnv;env=AMRBicycleEnv(get_map_spec('forest_a'),max_steps=600);viewer=TrainLiveViewer(enabled=True,fps=0,window_size=240,trail_len=20,skip_steps=1);obs,info=env.reset(seed=7);viewer.start_episode(env=env,env_name='forest_a',algo='cnn-ddqn',episode=1,total_episodes=1,info=info);pygame.event.post(pygame.event.Event(pygame.QUIT));obs,reward,done,truncated,info=env.step(0);viewer.update_step(env=env,env_name='forest_a',algo='cnn-ddqn',episode=1,total_episodes=1,step=1,action=0,reward=float(reward),done=bool(done),truncated=bool(truncated),info=info);print(f\"viewer_disabled={viewer.disabled}\");viewer.close()"`
- 输出要点：
  - `[train] live-view window closed by user; training continues`
  - `viewer_disabled=True`
- 结论：收到 `QUIT` 后 viewer 自动禁用，训练流程不抛异常（`PASS`）。
- 产物追踪：`run_dir/run_json/kpi = N/A`（该项为最小脚本级行为验证，不是完整 train/infer 任务）。

## 2026-02-10 碰撞检测框验证（live-view）

### CLI 参数检查
- 命令：`conda run -n ros2py310 python train.py --help`
- 结果：存在 `--live-view-collision-box | --no-live-view-collision-box`（`PASS`）。

### smoke（检测框开启）
- 命令：
  - `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_box_on`
- run_dir：`runs/repro_20260210_train_live_view_pygame_box_on/train_20260210_115024`
- 结果：`episodes_completed=1/1`，`stop_reason=completed`。
- run.json：`live_view_collision_box=true`。

### smoke（检测框关闭）
- 命令：
  - `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --no-live-view-collision-box --out repro_20260210_train_live_view_pygame_box_off`
- run_dir：`runs/repro_20260210_train_live_view_pygame_box_off/train_20260210_115102`
- 结果：`episodes_completed=1/1`，`stop_reason=completed`。
- run.json：`live_view_collision_box=false`。

### KPI / failure_reason
- 本轮为 train 行为验证，未执行 infer：
  - `short/long KPI`: `N/A`
  - `failure_reason`: `N/A`



## 2026-02-10 OBB 碰撞框验证（车体外框 + 航向角）

### smoke（OBB 开启）
- 命令：
  - `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_obb_on`
- run_dir：`runs/repro_20260210_train_live_view_pygame_obb_on/train_20260210_122251`
- 结果：`episodes_completed=1/1`，`stop_reason=completed`。
- run.json：`live_view_collision_box=true`。

### smoke（OBB 关闭）
- 命令：
  - `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --no-live-view-collision-box --out repro_20260210_train_live_view_pygame_obb_off`
- run_dir：`runs/repro_20260210_train_live_view_pygame_obb_off/train_20260210_122326`
- 结果：`episodes_completed=1/1`，`stop_reason=completed`。
- run.json：`live_view_collision_box=false`。

### 可视化结论
- 代码口径：碰撞框为固定车体外框（OBB），由 `pose_m` 航向角实时旋转。
- GUI 肉眼确认：`N/A`（本轮在 headless 环境执行）。

### KPI / failure_reason
- 本轮为 train 行为验证，未执行 infer：
  - `short/long KPI`: `N/A`
  - `failure_reason`: `N/A`
