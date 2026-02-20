# v7p2p2 runs 追溯

## 1. 本轮命令
- 远端自检：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端训练（smoke150）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2p2 --episodes 150 --out v7p2p2_smoke150 --device cuda --progress"`
- 远端推理（v7p2p2，runs=3）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2p2 --models v7p2p2_smoke150 --runs 3 --out v7p2p2_smoke150 --progress"`
- 远端对照推理（v7p1，runs=3）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p1 --models v7p1_remote150 --runs 3 --out v7p1_remote150_eval3 --progress"`

## 2. run 路径登记
- train（v7p2p2 smoke150）：
  - `run_dir`：`runs/v7p2p2_smoke150/train_20260220_230753`
  - `run.json`：`runs/v7p2p2_smoke150/train_20260220_230753/configs/run.json`
  - `train_meta`：`runs/v7p2p2_smoke150/train_20260220_230753/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2p2_smoke150/train_20260220_230753/train_flow.log`
- infer（v7p2p2，runs=3）：
  - `run_dir`：`runs/v7p2p2_smoke150/train_20260220_230753/infer/20260220_232053`
  - `run.json`：`runs/v7p2p2_smoke150/train_20260220_230753/infer/20260220_232053/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p2p2_smoke150/train_20260220_230753/infer/20260220_232053/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p2p2_smoke150/train_20260220_230753/infer/20260220_232053/table2_kpis_raw.csv`
- infer（v7p1 对照，runs=3）：
  - `run_dir`：`runs/v7p1_remote150_eval3/20260220_232121`
  - `run.json`：`runs/v7p1_remote150_eval3/20260220_232121/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p1_remote150_eval3/20260220_232121/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p1_remote150_eval3/20260220_232121/table2_kpis_raw.csv`

## 3. 结果登记
- train（v7p2p2 smoke150）：
  - `chosen_ckpt=best`
  - `episodes_completed=150`，`stop_reason=completed`
- infer（v7p2p2，CNN-DDQN）：
  - short：`success_rate=0.667`，`avg_path_length=16.1713`，`path_time_s=12.3750`
  - mid：`success_rate=0.667`，`avg_path_length=29.4152`，`path_time_s=23.5000`
  - long：`success_rate=0.333`，`avg_path_length=66.6176`，`path_time_s=44.2500`
- infer（v7p1 对照，CNN-DDQN）：
  - short：`success_rate=1.000`，`avg_path_length=16.7909`，`path_time_s=15.6667`
  - mid：`success_rate=0.667`，`avg_path_length=26.8669`，`path_time_s=23.5750`
  - long：`success_rate=0.333`，`avg_path_length=43.5187`，`path_time_s=28.1500`

## 4. 结论口径
- `v7p2p2`（ε 衰减修复）机制上生效，但 smoke 指标未体现稳定收益。
- 本版按失败归档处理，主线保持 `v7p1`。
