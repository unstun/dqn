# v7p2p5 runs 追溯

## 1) 本轮执行命令
- 本地 -> 远端同步（远端优先执行口径）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端（ubuntu-zt）self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke --models v7p2p5_globalcnn_fusionnorm_smoke --out v7p2p5_globalcnn_fusionnorm_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p5_globalcnn_fusionnorm_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p5_globalcnn_fusionnorm_smoke/`
- 本地单测：
  - `conda run -n ros2py310 python -m pytest tests/test_globalcnn_network.py -v`

## 2) run 路径登记
- train：
  - `run_dir`：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023`
  - `run_json`：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/configs/run.json`
  - `train_meta`：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626`
  - `run_json`：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626/table2_kpis_raw.csv`

## 3) 关键参数快照
- 复现配置：`configs/repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke.json`
- 关键差异参数：
  - `train.cnn_backbone=globalcnn_fusion`
  - `train.cnn_global_spatial_prior=true`
  - `train.cnn_global_prior_sigma=0.2`
  - `train.cnn_fusion_layernorm=true`
  - `train.cnn_fusion_layernorm_eps=1e-5`
- 推理关键参数（来自 infer `run.json`）：
  - `profile=repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke`
  - `runs=3`
  - `envs=[forest_a::short, forest_a::mid, forest_a::long]`
  - `baselines=[hybrid_astar_mpc]`

## 4) short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=0.333`，`avg_path_length=31.5744`，`path_time_s=17.1500`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- long：
  - CNN-DDQN：`success_rate=0.667`，`avg_path_length=89.4701`，`path_time_s=51.0000`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 5) `failure_reason` 分布
- CNN-DDQN：
  - short：`reached=1`，`timeout=2`
  - mid：`reached=2`，`timeout=1`
  - long：`reached=2`，`timeout=1`
  - 合计：`reached=5`，`timeout=4`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`
