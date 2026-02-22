# v7p3p3 runs 追溯

## 1) 本轮执行命令（实际）
- 本地 -> 远端同步：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260222_v7p3p3_infergate_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p3_infergate_smoke --models v7p3p3_infergate_smoke --out v7p3p3_infergate_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p3_infergate_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3p3_infergate_smoke/`

## 2) run 路径登记
- train：
  - `run_dir`：`runs/v7p3p3_infergate_smoke/train_20260222_112955`
  - `run_json`：`runs/v7p3p3_infergate_smoke/train_20260222_112955/configs/run.json`
  - `train_meta`：`runs/v7p3p3_infergate_smoke/train_20260222_112955/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p3p3_infergate_smoke/train_20260222_112955/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657`
  - `run_json`：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657/table2_kpis_raw.csv`

## 3) 关键参数快照
- 复现配置：`configs/repro_20260222_v7p3p3_infergate_smoke.json`
- 相对 `v7p3p2` 的关键差异（训练/推理一致）：
  - `forest_topk_turn_penalty=0.3`
  - `forest_min_progress_m=0.0`
- 训练元信息（`train_meta_forest_a.json` / `algos.cnn-ddqn`）：
  - `chosen_ckpt=best`
  - `stop_reason=rl_early_stop_plateau`
  - `forest_topk_turn_penalty=0.3`
  - `adaptive_no_progress_penalty_enabled=true`
  - `adaptive_no_progress_dist_gain=0.1`
  - `active_no_progress_penalty_mean=0.38277858664016956`
  - `active_no_progress_penalty_min=0.36225943235914504`
  - `active_no_progress_penalty_max=0.41773929990262454`
  - `adaptive_dist_ratio_mean=0.3277858664016966`

## 4) short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=0.000`（`avg_path_length/path_time_s=N/A`）
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- long：
  - CNN-DDQN：`success_rate=0.667`，`avg_path_length=61.6431`，`path_time_s=32.4000`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 5) `failure_reason` 分布
- CNN-DDQN：
  - short：`collision=1`，`timeout=2`
  - mid：`reached=1`，`collision=1`，`timeout=1`
  - long：`reached=2`，`timeout=1`
  - 合计：`reached=3`，`timeout=4`，`collision=2`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

