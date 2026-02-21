# v7p2p10 runs 追溯

## 1) 本轮执行命令
- 本地 -> 远端同步：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p10_penalty035_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p10_penalty035_smoke --models v7p2p10_penalty035_smoke --out v7p2p10_penalty035_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p10_penalty035_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p10_penalty035_smoke/`

## 2) run 路径登记
- train：
  - `run_dir`：`runs/v7p2p10_penalty035_smoke/train_20260221_234022`
  - `run_json`：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/configs/run.json`
  - `train_meta`：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340`
  - `run_json`：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340/table2_kpis_raw.csv`

## 3) 关键参数快照
- 复现配置：`configs/repro_20260221_v7p2p10_penalty035_smoke.json`
- 相对 `v7p2p9` 单变量改动：
  - `train.forest_reward_no_progress_penalty: 0.45 -> 0.35`
- 保持不变的关键参数：
  - `train.forest_expert_exploration=false`
  - `train.forest_train_dynamic_curriculum=true`
  - `train.grad_clip_norm=10.0`
  - `train.reward_scale=0.1`
  - `train.reward_clip_abs=10.0`
  - `profile=repro_20260221_v7p2p10_penalty035_smoke`
  - `runs=3`
  - `envs=[forest_a::short, forest_a::mid, forest_a::long]`
  - `baselines=[hybrid_astar_mpc]`

## 4) short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=0.667`，`avg_path_length=26.7961`，`path_time_s=25.0500`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- long：
  - CNN-DDQN：`success_rate=0.333`，`avg_path_length=51.0399`，`path_time_s=27.3500`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 5) `failure_reason` 分布
- CNN-DDQN：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=1`，`timeout=2`
  - long：`reached=1`，`timeout=1`，`collision=1`
  - 合计：`reached=4`，`timeout=4`，`collision=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`
