# v7p2p3 runs 追溯

## 1. 本轮命令
- 远端训练（allow early-stop, best）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2p3 --episodes 300 --out v7p2p3_train300_esbest --device cuda --progress --save-ckpt best"`
- 远端推理（best）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2p3 --models v7p2p3_train300_esbest --out v7p2p3_train300_esbest --progress"`

## 2. run 路径登记
- train：
  - `run_dir`：`runs/v7p2p3_train300_esbest/train_20260221_003108`
  - `run.json`：`runs/v7p2p3_train300_esbest/train_20260221_003108/configs/run.json`
  - `train_meta`：`runs/v7p2p3_train300_esbest/train_20260221_003108/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2p3_train300_esbest/train_20260221_003108/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p2p3_train300_esbest/train_20260221_003108/infer/20260221_004529`
  - `run.json`：`runs/v7p2p3_train300_esbest/train_20260221_003108/infer/20260221_004529/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p2p3_train300_esbest/train_20260221_003108/infer/20260221_004529/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p2p3_train300_esbest/train_20260221_003108/infer/20260221_004529/table2_kpis_raw.csv`

## 3. 结果登记
- train：
  - `episodes_completed=220/300`
  - `stop_reason=rl_early_stop_plateau`
  - `chosen_ckpt=best`
- infer（CNN-DDQN）：
  - short：`success_rate=0.600`
  - mid：`success_rate=0.200`
  - long：`success_rate=0.600`

## 4. 结论口径
- 版本目标（ε 衰减修复）在机制上达成。
- 在本轮 train300 + infer(runs=5) 下，指标不优于 `v7p1_train300_esbest`，本版归档为失败尝试。
