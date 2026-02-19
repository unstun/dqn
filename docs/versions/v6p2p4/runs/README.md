# v6p2p4 runs 追溯

## 1. 本轮命令
- 训练：
  - `conda run -n ros2py310 python train.py --profile v6p2p4`
- 推理：
  - `conda run -n ros2py310 python infer.py --profile v6p2p4`

## 2. run 路径登记
- train `run_dir`：`runs/v6p2p4/train_20260219_153029`
- train `run.json`：`runs/v6p2p4/train_20260219_153029/configs/run.json`
- train `train_meta`：`runs/v6p2p4/train_20260219_153029/configs/train_meta_forest_a.json`
- infer `run_dir`：`runs/v6p2p4/train_20260219_153029/infer/20260219_161252`
- infer `run.json`：`runs/v6p2p4/train_20260219_153029/infer/20260219_161252/configs/run.json`
- KPI（均值）：`runs/v6p2p4/train_20260219_153029/infer/20260219_161252/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v6p2p4/train_20260219_153029/infer/20260219_161252/table2_kpis_raw.csv`

## 3. 结果登记
- 训练：
  - `cnn-ddqn`：`episodes=300/300`，`stop_reason=completed`
  - `ddpg`：`episodes=300/300`，`stop_reason=completed`
  - `sac`：`episodes=300/300`，`stop_reason=completed`
- 推理（runs=5）：
  - short：
    - CNN-DDQN `success_rate=0.80`，DDPG `0.00`，SAC `0.00`，Hybrid `1.00`
  - mid：
    - CNN-DDQN `success_rate=0.80`，DDPG `0.00`，SAC `0.00`，Hybrid `1.00`
  - long：
    - CNN-DDQN `success_rate=1.00`，DDPG `0.00`，SAC `0.00`，Hybrid `1.00`

## 4. 备注
- 本版新增连续控制对比算法 `DDPG/SAC`，并保持与 `v6p2p3` 相同的 train/infer unified 规则口径。
- 连续算法推理阶段失败以 `collision/timeout` 为主，需后续做奖励与动作约束修订后再评估。
