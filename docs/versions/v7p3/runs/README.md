# v7p3 runs 追溯

## 1) 本轮执行命令（实际）
- 本地 -> 远端同步：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p3_suite_penalty_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p3_suite_penalty_smoke --models v7p3_suite_penalty_smoke --out v7p3_suite_penalty_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3_suite_penalty_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3_suite_penalty_smoke/`

## 2) run 路径登记
- train：
  - `run_dir`：`runs/v7p3_suite_penalty_smoke/train_20260222_012415`
  - `run_json`：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/configs/run.json`
  - `train_meta`：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023`
  - `run_json`：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023/table2_kpis_raw.csv`

## 3) 关键参数快照
- 复现配置：`configs/repro_20260221_v7p3_suite_penalty_smoke.json`
- 相对 `v7p2p10` 的关键差异：
  - `train.forest_reward_no_progress_penalty=0.40`
  - `train.forest_train_suite_no_progress_penalty=true`
  - `train.forest_train_short_no_progress_penalty=0.45`
  - `train.forest_train_long_no_progress_penalty=0.35`
- 训练元信息（`train_meta_forest_a.json`）：
  - `suite_no_progress_penalty_enabled=true`
  - `suite_no_progress_penalty_short=0.45`
  - `suite_no_progress_penalty_long=0.35`
  - `dynamic_short_prob_final=0.4875`

## 4) short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=0.667`，`avg_path_length=25.3610`，`path_time_s=17.4500`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- long：
  - CNN-DDQN：`success_rate=0.333`，`avg_path_length=57.1796`，`path_time_s=30.1500`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 5) `failure_reason` 分布
- CNN-DDQN：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=2`，`collision=1`
  - long：`reached=1`，`timeout=1`，`collision=1`
  - 合计：`reached=5`，`timeout=2`，`collision=2`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`
