# v7p3p1 runs 追溯

## 1) 本轮执行命令（实际）
- 本地 -> 远端同步：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260222_v7p3p1_adaptive_penalty_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p1_adaptive_penalty_smoke --models v7p3p1_adaptive_penalty_smoke --out v7p3p1_adaptive_penalty_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p1_adaptive_penalty_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3p1_adaptive_penalty_smoke/`

## 2) run 路径登记
- train：
  - `run_dir`：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303`
  - `run_json`：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/configs/run.json`
  - `train_meta`：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552`
  - `run_json`：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552/table2_kpis_raw.csv`

## 3) 关键参数快照
- 复现配置：`configs/repro_20260222_v7p3p1_adaptive_penalty_smoke.json`
- 相对 `v7p3` 的关键差异：
  - `train.forest_train_suite_no_progress_penalty=false`
  - `train.forest_train_adaptive_no_progress_penalty=true`
  - `train.forest_train_no_progress_penalty_dist_gain=0.15`
  - `train.forest_train_no_progress_penalty_min=0.35`
  - `train.forest_train_no_progress_penalty_max=0.50`
  - `train.forest_reward_no_progress_penalty=0.35`
- 训练元信息（`train_meta_forest_a.json`）：
  - `adaptive_no_progress_penalty_enabled=true`
  - `adaptive_no_progress_dist_gain=0.15`
  - `active_no_progress_penalty_mean=0.3886967415909185`
  - `active_no_progress_penalty_min=0.36817668807868204`
  - `active_no_progress_penalty_max=0.43840775404389337`
  - `adaptive_dist_ratio_mean=0.25797827727278994`

## 4) short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=0.667`，`avg_path_length=32.0322`，`path_time_s=24.3750`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- long：
  - CNN-DDQN：`success_rate=1.000`，`avg_path_length=83.8999`，`path_time_s=48.1000`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 5) `failure_reason` 分布
- CNN-DDQN：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=8`，`timeout=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`
