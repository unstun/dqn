# v7p3p7 runs 追溯

## 1) 本轮执行命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 train smoke：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke"`
- 远端 infer smoke：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke --models runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p7_obsmap128_timeoutcut_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3p7_obsmap128_timeoutcut_smoke/`

## 2) run 路径登记
- train：
  - `run_dir`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248`
  - `run_json`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/configs/run.json`
  - `train_meta`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/configs/train_meta_forest_a.json`
  - `train_flow`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/train_flow.log`
  - `returns_csv`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/training_returns.csv`
  - `eval_csv`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/training_eval.csv`
  - `model_ckpt`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/models/forest_a/cnn-ddqn.pt`
- infer：
  - `run_dir`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329`
  - `run_json`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329/configs/run.json`
  - `kpi_mean_raw`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329/table2_kpis_raw.csv`

## 3) 关键参数快照
- profile：`repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke`
- 关键参数：
  - `obs_map_size=128`
  - `replay_capacity=10000`
  - `batch_size=32`
  - `episodes=150`
  - `runs=3`
  - `forest_no_fallback=false`
  - `forest_topk=12`
  - `forest_topk_turn_penalty=0.2`
  - `forest_min_progress_m=0.02`
  - `forest_train_short_prob=0.20`
  - `forest_train_dynamic_target_sr_long=0.90`

## 4) 结果状态
- train：完成（`Run done: total_elapsed=49m56.9s`）
- infer：完成（`runs=3`，short/mid/long）
- KPI（CNN）：
  - short：`SR=1.000`，`avg_path_length=21.3138`，`path_time_s=13.9000`
  - mid：`SR=1.000`，`avg_path_length=36.6758`，`path_time_s=21.2500`
  - long：`SR=0.333`，`avg_path_length=66.8440`，`path_time_s=35.7500`
- `failure_reason`：
  - CNN-DDQN：`reached=7`，`timeout=2`
  - Hybrid A*-MPC：`reached=9`
- 版本结论：`NO-GO（失败归档）`
