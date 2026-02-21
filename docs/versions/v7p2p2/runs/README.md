# v7p2p2 runs 追溯

## 1. 本轮执行命令
- 本地 -> 远端同步（远端优先执行口径）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端（ubuntu-zt）self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p2_globalcnn_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p2_globalcnn_smoke --models v7p2p2_globalcnn_smoke --out v7p2p2_globalcnn_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p2_globalcnn_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p2_globalcnn_smoke/`
- 本地单测：
  - `conda run -n ros2py310 python -m pytest tests/test_globalcnn_network.py -v`

## 2. run 路径登记
- train：
  - `run_dir`：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611`
  - `run_json`：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/configs/run.json`
  - `train_meta`：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943`
  - `run_json`：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943/table2_kpis_raw.csv`

## 3. 关键参数快照
- 复现配置：`configs/repro_20260221_v7p2p2_globalcnn_smoke.json`
- 关键差异参数：
  - `train.cnn_backbone=globalcnn`
  - `train.cnn_global_width=32`
  - `train.cnn_global_dropout=0.1`
- 推理关键参数（来自 infer `run.json`）：
  - `profile=repro_20260221_v7p2p2_globalcnn_smoke`
  - `runs=3`
  - `envs=[forest_a::short, forest_a::mid, forest_a::long]`
  - `baselines=[hybrid_astar_mpc]`

## 4. short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=0.667`，`avg_path_length=15.8975`，`path_time_s=10.1250`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- long：
  - CNN-DDQN：`success_rate=0.333`，`avg_path_length=44.2185`，`path_time_s=28.3000`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 5. `failure_reason` 分布
- CNN-DDQN：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=1`，`collision=1`，`timeout=1`
  - long：`reached=1`，`timeout=2`
  - 合计：`reached=4`，`timeout=4`，`collision=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`
