# v6p2p3 runs 追溯

## 1. 本轮命令
- 训练（第一次，触发早停）：
  - `conda run -n ros2py310 python train.py --profile v6p2p3`
- 训练（第二次，强制跑满 300 轮）：
  - `conda run -n ros2py310 python train.py --profile v6p2p3 --rl-early-stop-patience-points 9999 --rl-early-stop-warmup-episodes 9999`
- 推理（short/mid/long 各 5 轮）：
  - `conda run -n ros2py310 python infer.py --profile v6p2p3`

## 2. run 路径登记
- train#1 `run_dir`：`runs/v6p2p3/train_20260219_135522`
- train#1 `run.json`：`runs/v6p2p3/train_20260219_135522/configs/run.json`
- train#1 `train_meta`：`runs/v6p2p3/train_20260219_135522/configs/train_meta_forest_a.json`
- train#2 `run_dir`：`runs/v6p2p3/train_20260219_142104`
- train#2 `run.json`：`runs/v6p2p3/train_20260219_142104/configs/run.json`
- train#2 `train_meta`：`runs/v6p2p3/train_20260219_142104/configs/train_meta_forest_a.json`
- infer `run_dir`：`runs/v6p2p3/train_20260219_142104/infer/20260219_145315`
- infer `run.json`：`runs/v6p2p3/train_20260219_142104/infer/20260219_145315/configs/run.json`
- KPI（均值）：`runs/v6p2p3/train_20260219_142104/infer/20260219_145315/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v6p2p3/train_20260219_142104/infer/20260219_145315/table2_kpis_raw.csv`

## 3. 结果登记
- train#1：
  - `episodes=170/300`，`stop_reason=rl_early_stop_plateau`
- train#2：
  - `episodes=300/300`，`stop_reason=completed`
- infer（runs=5）：
  - short（CNN）：`success_rate=0.80`，`avg_path_length=15.7615`，`path_time_s=9.3625`
  - mid（CNN）：`success_rate=1.00`，`avg_path_length=25.3193`，`path_time_s=15.0600`
  - long（CNN）：`success_rate=1.00`，`avg_path_length=46.2403`，`path_time_s=28.9500`
  - short（Hybrid）：`success_rate=1.00`，`avg_path_length=16.8724`，`path_time_s=10.0000`
  - mid（Hybrid）：`success_rate=1.00`，`avg_path_length=25.1525`，`path_time_s=13.8700`
  - long（Hybrid）：`success_rate=1.00`，`avg_path_length=43.0247`，`path_time_s=22.8200`
  - `failure_reason`（CNN 汇总）：`reached=14`, `collision=1`
  - `failure_reason`（Hybrid 汇总）：`reached=15`

## 4. 备注
- 代表模型应以 train#2（300 轮完成）为准。
- 本版重点是训练/推理规则统一（含 `forest_min_od_m`、`forest_topk`、`stop override` 与 `no_terminate_on_stuck` 口径统一）。
- 最终研究结论仍需 short/long 各 `runs=20`。
