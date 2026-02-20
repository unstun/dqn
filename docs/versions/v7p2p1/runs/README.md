# v7p2p1 runs 追溯

## 1. 本轮命令
- 远端训练（smoke150）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2 --episodes 150 --out v7p2_es150 --device cuda --progress"`
- 远端推理（runs=20）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_es150 --runs 20 --out v7p2_es150 --progress"`
- 远端推理（runs=3，对照 smoke）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_es150 --runs 3 --out v7p2_es150_eval3 --progress"`
- 对照尝试（失败）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p1 --models v7p1_remote150 --runs 20 --out v7p1_remote150_eval20 --progress"`
- 本地回退后兼容性抽检（成功）：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models v7p1_remote150 --envs forest_a::short --runs 1 --out rollback_v7p1_check --progress`

## 2. run 路径登记
- train（smoke150）：
  - `run_dir`：`runs/v7p2_es150/train_20260220_222056`
  - `run.json`：`runs/v7p2_es150/train_20260220_222056/configs/run.json`
  - `train_meta`：`runs/v7p2_es150/train_20260220_222056/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2_es150/train_20260220_222056/train_flow.log`
- infer（runs=20）：
  - `run_dir`：`runs/v7p2_es150/train_20260220_222056/infer/20260220_223016`
  - `run.json`：`runs/v7p2_es150/train_20260220_222056/infer/20260220_223016/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p2_es150/train_20260220_222056/infer/20260220_223016/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p2_es150/train_20260220_222056/infer/20260220_223016/table2_kpis_raw.csv`
- infer（runs=3）：
  - `run_dir`：`runs/v7p2_es150_eval3/20260220_223301`
  - `run.json`：`runs/v7p2_es150_eval3/20260220_223301/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p2_es150_eval3/20260220_223301/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p2_es150_eval3/20260220_223301/table2_kpis_raw.csv`

## 3. 结果登记
- train（smoke150）：
  - `chosen_ckpt=best`
  - `episodes_completed=150`，`stop_reason=completed`
- infer（runs=20，CNN-DDQN）：
  - short：`success_rate=0.85`，`avg_path_length=23.3004`，`path_time_s=16.1235`
  - mid：`success_rate=0.80`，`avg_path_length=28.7948`，`path_time_s=18.0438`
  - long：`success_rate=0.65`，`avg_path_length=66.0207`，`path_time_s=41.4731`
- infer（runs=3，CNN-DDQN）：
  - short：`success_rate=1.00`
  - mid：`success_rate=0.333`
  - long：`success_rate=1.00`

## 4. 失败记录（必须归档）
- `v7p1_remote150` 的 `runs=20` 对照推理失败（本轮未生成 `run_dir`）：
  - `run_dir`：`N/A`
  - `run.json`：`N/A`
  - 原因：checkpoint 与当前环境观测维度不一致。
  - 报错：`Checkpoint expects obs_dim=154 but env provides obs_dim=155 for 'forest_a'. Re-train models to match the environment observation space.`

## 5. 结论口径
- 本版定义为失败版本 `v7p2p1`。
- 主线回退到 `v7p1`，后续从 `v7p2p2` 继续迭代。

## 6. 回退验证记录
- 回退后 `v7p1_remote150` 已可在当前代码口径下正常推理（1 轮抽检通过）。
- 抽检 run：
  - `run_dir`：`runs/rollback_v7p1_check/20260220_045505`
  - `kpi_mean_raw`：`runs/rollback_v7p1_check/20260220_045505/table2_kpis_mean_raw.csv`
