# v6p2p5 runs 追溯

## 1. 本轮命令（smoke）
- 训练（120 轮）：
  - `conda run -n ros2py310 python train.py --profile v6p2p5 --episodes 120 --out v6p2p5_smoke120`
- 推理（short/mid/long 各 runs=3）：
  - `conda run -n ros2py310 python infer.py --profile v6p2p5 --models runs/v6p2p5_smoke120/train_20260219_030737/models --runs 3 --out v6p2p5_smoke120`

## 2. run 路径登记
- train `run_dir`：`runs/v6p2p5_smoke120/train_20260219_030737`
- infer `run_dir`：`runs/v6p2p5_smoke120/train_20260219_030737/infer/20260219_033302`
- train `run.json`：`runs/v6p2p5_smoke120/train_20260219_030737/configs/run.json`
- infer `run.json`：`runs/v6p2p5_smoke120/train_20260219_030737/infer/20260219_033302/configs/run.json`
- KPI（均值）：`runs/v6p2p5_smoke120/train_20260219_030737/infer/20260219_033302/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v6p2p5_smoke120/train_20260219_030737/infer/20260219_033302/table2_kpis_raw.csv`
- 训练元信息：`runs/v6p2p5_smoke120/train_20260219_030737/configs/train_meta_forest_a.json`

## 3. 结果登记
- smoke（三套件合并）：
  - `CNN-DDQN`：`success_rate=0.889`（`reached=8, timeout=1`）
  - `DDPG`：`success_rate=0.111`（`reached=1, timeout=8`）
  - `SAC`：`success_rate=0.000`（`timeout=9`）
  - `Hybrid A*-MPC`：`success_rate=1.000`（`reached=9`）
- 分套件关键 SR：
  - short：`CNN-DDQN=1.000, DDPG=0.333, SAC=0.000, Hybrid A*-MPC=1.000`
  - mid：`CNN-DDQN=0.667, DDPG=0.000, SAC=0.000, Hybrid A*-MPC=1.000`
  - long：`CNN-DDQN=1.000, DDPG=0.000, SAC=0.000, Hybrid A*-MPC=1.000`

## 4. 备注
- 训练阶段 `training_eval.xlsx` 因环境缺少 `openpyxl` 未写出，但 `training_eval.csv` 已正常生成。
- 本次为 smoke 口径；最终结论仍需 `short/long + runs=20`。
