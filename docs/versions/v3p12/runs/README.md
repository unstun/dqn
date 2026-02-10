# v3p12 - 运行记录

## 代表 run（smoke）
- `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a_fast/train_20260210_020135`
  - run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a_fast/train_20260210_020135/configs/run.json`
  - train_meta: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a_fast/train_20260210_020135/configs/train_meta_forest_a.json`
  - train_eval: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a_fast/train_20260210_020135/training_eval.csv`
- `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b_fast/train_20260210_021049`
  - run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b_fast/train_20260210_021049/configs/run.json`
  - train_meta: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b_fast/train_20260210_021049/configs/train_meta_forest_a.json`
  - train_eval: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b_fast/train_20260210_021049/training_eval.csv`
- `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast/train_20260210_021930`
  - run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast/train_20260210_021930/configs/run.json`
  - train_meta: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast/train_20260210_021930/configs/train_meta_forest_a.json`
  - train_eval: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast/train_20260210_021930/training_eval.csv`

## 代表 run（full）
- `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622`
  - run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622/configs/run.json`
  - kpi_mean_raw: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622/table2_kpis_mean_raw.csv`
  - kpi_raw: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622/table2_kpis_raw.csv`

## 熔断验证 run
- `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_throughput_test/train_20260210_015016`
  - run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_throughput_test/train_20260210_015016/configs/run.json`
  - train_meta: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_throughput_test/train_20260210_015016/configs/train_meta_forest_a.json`
  - 说明：failure_reason=`training_throughput_abnormal`

## 失败运行（N/A）
- `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b/train_20260210_023020`
  - run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_b/train_20260210_023020/configs/run.json`
  - kpi: `N/A`
  - 原因：`manual_stop_time_budget`
- `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a/train_20260210_014204`
  - run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_a/train_20260210_014204/configs/run.json`
  - kpi: `N/A`
  - 原因：`manual_stop_time_budget`

## 聚合文件
- `runs/v3p12_smoke_full_summary.csv`
- `runs/v3p12_smoke_full_failure_dist.csv`
- `runs/v3p12_train_progress_meta_summary.csv`

## 2026-02-10 训练实时显示 smoke 运行
- `runs/repro_20260210_train_live_view_pygame_smoke_headless/train_20260210_113131`
  - command: `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_smoke_headless`
  - run_json: `runs/repro_20260210_train_live_view_pygame_smoke_headless/train_20260210_113131/configs/run.json`
  - train_meta: `runs/repro_20260210_train_live_view_pygame_smoke_headless/train_20260210_113131/configs/train_meta_forest_a.json`
  - training_returns: `runs/repro_20260210_train_live_view_pygame_smoke_headless/train_20260210_113131/training_returns.csv`
  - kpi: `N/A`（本轮仅训练 smoke，未执行 infer 评测）

- `runs/repro_20260210_train_live_view_pygame_smoke_noview/train_20260210_113351`
  - command: `conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --no-live-view --out repro_20260210_train_live_view_pygame_smoke_noview`
  - run_json: `runs/repro_20260210_train_live_view_pygame_smoke_noview/train_20260210_113351/configs/run.json`
  - train_meta: `runs/repro_20260210_train_live_view_pygame_smoke_noview/train_20260210_113351/configs/train_meta_forest_a.json`
  - training_returns: `runs/repro_20260210_train_live_view_pygame_smoke_noview/train_20260210_113351/training_returns.csv`
  - kpi: `N/A`（本轮仅训练 smoke，未执行 infer 评测）

## 2026-02-10 失败运行（N/A）
- command: `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 120 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_smoke_headless`
- run_dir: `N/A`
- run_json: `N/A`
- kpi: `N/A`
- 原因：`max_steps_too_small_for_forest_env`（`max_steps=120` 小于环境要求 `511`）。

## 2026-02-10 GUI 手工验证记录
- 计划命令：`conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --live-view --live-view-fps 0 --live-view-window-size 900 --live-view-trail-len 300 --live-view-skip-steps 1`
- run_dir: `N/A`
- run_json: `N/A`
- kpi: `N/A`
- 原因：当前轮次在 headless 环境，未进行桌面窗口人工验证。



## 2026-02-10 二次验证（已安装 pygame）
- `runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849`
  - command: `conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_gui_try`
  - run_json: `runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849/configs/run.json`
  - train_meta: `runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849/configs/train_meta_forest_a.json`
  - training_returns: `runs/repro_20260210_train_live_view_pygame_gui_try/train_20260210_113849/training_returns.csv`
  - kpi: `N/A`（本轮仅 train smoke，未执行 infer）

- 自动关窗脚本验证
  - command: `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python -c "import sys;from pathlib import Path;sys.path.insert(0,str(Path('/home/sun/phdproject/dqn/dqn')));import pygame;from forest_vehicle_dqn.live_view_pygame import TrainLiveViewer;from forest_vehicle_dqn.maps import get_map_spec;from forest_vehicle_dqn.env import AMRBicycleEnv;env=AMRBicycleEnv(get_map_spec('forest_a'),max_steps=600);viewer=TrainLiveViewer(enabled=True,fps=0,window_size=240,trail_len=20,skip_steps=1);obs,info=env.reset(seed=7);viewer.start_episode(env=env,env_name='forest_a',algo='cnn-ddqn',episode=1,total_episodes=1,info=info);pygame.event.post(pygame.event.Event(pygame.QUIT));obs,reward,done,truncated,info=env.step(0);viewer.update_step(env=env,env_name='forest_a',algo='cnn-ddqn',episode=1,total_episodes=1,step=1,action=0,reward=float(reward),done=bool(done),truncated=bool(truncated),info=info);print(f\"viewer_disabled={viewer.disabled}\");viewer.close()"`
  - run_dir: `N/A`
  - run_json: `N/A`
  - kpi: `N/A`
  - reason: 行为级脚本验证（非完整 train/infer run）。

## 2026-02-10 live-view 碰撞检测框 runs
- `runs/repro_20260210_train_live_view_pygame_box_on/train_20260210_115024`
  - command: `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_box_on`
  - run_json: `runs/repro_20260210_train_live_view_pygame_box_on/train_20260210_115024/configs/run.json`
  - train_meta: `runs/repro_20260210_train_live_view_pygame_box_on/train_20260210_115024/configs/train_meta_forest_a.json`
  - training_returns: `runs/repro_20260210_train_live_view_pygame_box_on/train_20260210_115024/training_returns.csv`
  - kpi: `N/A`（仅 train smoke）

- `runs/repro_20260210_train_live_view_pygame_box_off/train_20260210_115102`
  - command: `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --no-live-view-collision-box --out repro_20260210_train_live_view_pygame_box_off`
  - run_json: `runs/repro_20260210_train_live_view_pygame_box_off/train_20260210_115102/configs/run.json`
  - train_meta: `runs/repro_20260210_train_live_view_pygame_box_off/train_20260210_115102/configs/train_meta_forest_a.json`
  - training_returns: `runs/repro_20260210_train_live_view_pygame_box_off/train_20260210_115102/training_returns.csv`
  - kpi: `N/A`（仅 train smoke）



## 2026-02-10 OBB 碰撞框 runs
- `runs/repro_20260210_train_live_view_pygame_obb_on/train_20260210_122251`
  - command: `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --out repro_20260210_train_live_view_pygame_obb_on`
  - run_json: `runs/repro_20260210_train_live_view_pygame_obb_on/train_20260210_122251/configs/run.json`
  - train_meta: `runs/repro_20260210_train_live_view_pygame_obb_on/train_20260210_122251/configs/train_meta_forest_a.json`
  - training_returns: `runs/repro_20260210_train_live_view_pygame_obb_on/train_20260210_122251/training_returns.csv`
  - kpi: `N/A`（仅 train smoke）

- `runs/repro_20260210_train_live_view_pygame_obb_off/train_20260210_122326`
  - command: `SDL_VIDEODRIVER=dummy conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --episodes 1 --max-steps 600 --eval-every 0 --train-eval-every 0 --no-live-view-collision-box --out repro_20260210_train_live_view_pygame_obb_off`
  - run_json: `runs/repro_20260210_train_live_view_pygame_obb_off/train_20260210_122326/configs/run.json`
  - train_meta: `runs/repro_20260210_train_live_view_pygame_obb_off/train_20260210_122326/configs/train_meta_forest_a.json`
  - training_returns: `runs/repro_20260210_train_live_view_pygame_obb_off/train_20260210_122326/training_returns.csv`
  - kpi: `N/A`（仅 train smoke）
