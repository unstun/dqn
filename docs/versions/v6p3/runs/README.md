# v6p3 runs 追溯

## 1. 本轮命令
- self-check（已执行并通过）：
  - `conda run -n ros2py310 python train.py --profile v6p3 --self-check`
  - `conda run -n ros2py310 python infer.py --profile v6p3 --self-check`
- smoke（已执行）：
  - `conda run -n ros2py310 python train.py --profile v6p3 --episodes 120 --out v6p3_smoke120`
  - `conda run -n ros2py310 python infer.py --profile v6p3 --models runs/v6p3_smoke120/train_20260219_041557/models --runs 3 --out v6p3_smoke120`
- full（待执行）：
  - `conda run -n ros2py310 python train.py --profile v6p3 --episodes 300 --out v6p3_full300`
  - `conda run -n ros2py310 python infer.py --profile v6p3 --models v6p3_full300 --runs 20 --out v6p3_full300`

## 2. run 路径登记
- train `run_dir`：`runs/v6p3_smoke120/train_20260219_041557`
- infer `run_dir`：`runs/v6p3_smoke120/train_20260219_041557/infer/20260219_043629`
- train `run.json`：`runs/v6p3_smoke120/train_20260219_041557/configs/run.json`
- infer `run.json`：`runs/v6p3_smoke120/train_20260219_041557/infer/20260219_043629/configs/run.json`
- KPI（均值）：`runs/v6p3_smoke120/train_20260219_041557/infer/20260219_043629/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v6p3_smoke120/train_20260219_041557/infer/20260219_043629/table2_kpis_raw.csv`
- 训练元信息：`runs/v6p3_smoke120/train_20260219_041557/configs/train_meta_forest_a.json`

## 3. 结果登记
- self-check：
  - `train.py --profile v6p3 --self-check`：通过（CUDA 设备可用）
  - `infer.py --profile v6p3 --self-check`：通过（CUDA 架构校验通过）
- smoke（short/mid/long 合并）：
  - `CNN-DDQN`：`success_rate=0.778`（`reached=7, collision=2`）
  - `Hybrid A*-MPC`：`success_rate=1.000`（`reached=9`）
- 分套件关键 SR：
  - short：`CNN-DDQN=1.000, Hybrid A*-MPC=1.000`
  - mid：`CNN-DDQN=0.333, Hybrid A*-MPC=1.000`
  - long：`CNN-DDQN=1.000, Hybrid A*-MPC=1.000`
- full：
  - `N/A`（待运行）

## 4. 备注
- 本版本只保留 `cnn-ddqn` 与 `Hybrid A*-MPC` 对比口径。
- 每次执行 train/infer 后，必须同步回填本文件与 `RESULTS.md`。
- 训练阶段 `training_eval.xlsx` 因环境缺少 `openpyxl` 未写出，但 `training_eval.csv` 已正常生成。
