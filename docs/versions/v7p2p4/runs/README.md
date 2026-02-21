# v7p2p4 runs 追溯

## 1) 本轮执行命令
- 本地 -> 远端同步（远端优先执行口径）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端（ubuntu-zt）self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p4_globalcnn_spatialprior_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p4_globalcnn_spatialprior_smoke --models v7p2p4_globalcnn_spatialprior_smoke --out v7p2p4_globalcnn_spatialprior_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p4_globalcnn_spatialprior_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p4_globalcnn_spatialprior_smoke/`
- 本地单测：
  - `conda run -n ros2py310 python -m pytest tests/test_globalcnn_network.py -v`

## 2) run 路径登记
- train：
  - `run_dir`：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908`
  - `run_json`：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/configs/run.json`
  - `train_meta`：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926`
  - `run_json`：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926/table2_kpis_raw.csv`

## 3) 关键参数快照
- 复现配置：`configs/repro_20260221_v7p2p4_globalcnn_spatialprior_smoke.json`
- 关键差异参数：
  - `train.cnn_backbone=globalcnn_fusion`
  - `train.cnn_global_spatial_prior=true`
  - `train.cnn_global_prior_sigma=0.2`
  - `train.cnn_global_width=32`
  - `train.cnn_global_dropout=0.1`
- 推理关键参数（来自 infer `run.json`）：
  - `profile=repro_20260221_v7p2p4_globalcnn_spatialprior_smoke`
  - `runs=3`
  - `envs=[forest_a::short, forest_a::mid, forest_a::long]`
  - `baselines=[hybrid_astar_mpc]`

## 4) short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=0.667`，`avg_path_length=17.9810`，`path_time_s=10.7250`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- long：
  - CNN-DDQN：`success_rate=1.000`，`avg_path_length=51.1817`，`path_time_s=28.2000`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 5) `failure_reason` 分布
- CNN-DDQN：
  - short：`collision=1`，`reached=2`
  - mid：`reached=2`，`timeout=1`
  - long：`reached=3`
  - 合计：`reached=7`，`timeout=1`，`collision=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`
