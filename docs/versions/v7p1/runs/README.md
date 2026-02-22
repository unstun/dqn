# v7p1 runs 追溯

## 1. 本轮命令
- 训练：
  - `conda run -n ros2py310 python train.py --profile v7p1 --episodes 300 --out v7p1_train300_esbest --device cuda --progress --save-ckpt best`
- 推理：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models v7p1_train300_esbest --out v7p1_train300_esbest --progress`
  - 说明：profile 默认 `runs=5`。

## 2. run 路径登记
- train：
  - `run_dir`：`runs/v7p1_train300_esbest/train_20260221_010743`
  - `run_json`：`runs/v7p1_train300_esbest/train_20260221_010743/configs/run.json`
  - `train_meta`：`runs/v7p1_train300_esbest/train_20260221_010743/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p1_train300_esbest/train_20260221_010743/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927`
  - `run_json`：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_raw.csv`

## 3. 关键参数快照（run.json）
- train：
  - `profile=v7p1`，`episodes=300`，`save_ckpt=best`，`device=cuda`
  - `forest_no_fallback=false`，`forest_topk=10`，`forest_adm_horizon=30`
  - `forest_reward_k_t=0.10`，`forest_reward_k_delta=0.8`
- infer：
  - `profile=v7p1`，`models=v7p1_train300_esbest`，`runs=5`
  - `envs=[forest_a::short, forest_a::mid, forest_a::long]`
  - `baselines=[hybrid_astar_mpc]`
  - `forest_no_fallback=false`，`forest_topk=10`，`forest_adm_horizon=30`

## 4. short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=16.7849`，`path_time_s=11.60`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=16.8724`，`path_time_s=10.00`
- long：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=51.9081`，`path_time_s=34.01`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=43.0247`，`path_time_s=22.82`

## 5. `failure_reason` 分布
- CNN-DDQN：
  - short：`reached=5`
  - mid：`reached=5`
  - long：`reached=5`
- Hybrid A*-MPC：
  - short：`reached=5`
  - mid：`reached=5`
  - long：`reached=5`

## 6. 附：v8 推理期消融（基于 v7p1 checkpoint，infer-only）

> 目的：在保持 `success_rate≈1.0` 前提下，先验证推理期 admissible gating（可采纳动作判定）的参数敏感性，为后续 v8 训练侧改动提供证据。

- 固定模型：`runs/v7p1_train300_esbest/train_20260221_010743`
- 固定 profile：`v7p1`
- 固定口径：`runs=3`，`baselines=[hybrid_astar_mpc]`，`random_start_goal=true`
- 说明：本轮 R0–R4 的 baseline（Hybrid A*-MPC）均值 KPI 行逐项一致（见各 `table2_kpis_mean_raw.csv`），因此可直接横向对比 RL 指标。

### R0（对照）
- 命令：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models runs/v7p1_train300_esbest/train_20260221_010743 --out v8_ablate_v7p1_R0_h30_mp001 --runs 3 --forest-adm-horizon 30 --forest-min-progress-m 0.01`
- infer：
  - `run_dir`：`runs/v8_ablate_v7p1_R0_h30_mp001/20260223_004917`
  - `run_json`：`runs/v8_ablate_v7p1_R0_h30_mp001/20260223_004917/configs/run.json`
  - `kpi_mean_raw`：`runs/v8_ablate_v7p1_R0_h30_mp001/20260223_004917/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v8_ablate_v7p1_R0_h30_mp001/20260223_004917/table2_kpis_raw.csv`

### R1（放宽进度阈值：`min_progress_m=0.0`）
- 命令：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models runs/v7p1_train300_esbest/train_20260221_010743 --out v8_ablate_v7p1_R1_h30_mp000 --runs 3 --forest-adm-horizon 30 --forest-min-progress-m 0.0`
- infer：
  - `run_dir`：`runs/v8_ablate_v7p1_R1_h30_mp000/20260223_005010`
  - `run_json`：`runs/v8_ablate_v7p1_R1_h30_mp000/20260223_005010/configs/run.json`
  - `kpi_mean_raw`：`runs/v8_ablate_v7p1_R1_h30_mp000/20260223_005010/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v8_ablate_v7p1_R1_h30_mp000/20260223_005010/table2_kpis_raw.csv`

### R2（缩短 admissible horizon：`adm_h=15`）
- 命令：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models runs/v7p1_train300_esbest/train_20260221_010743 --out v8_ablate_v7p1_R2_h15_mp001 --runs 3 --forest-adm-horizon 15 --forest-min-progress-m 0.01`
- infer：
  - `run_dir`：`runs/v8_ablate_v7p1_R2_h15_mp001/20260223_005037`
  - `run_json`：`runs/v8_ablate_v7p1_R2_h15_mp001/20260223_005037/configs/run.json`
  - `kpi_mean_raw`：`runs/v8_ablate_v7p1_R2_h15_mp001/20260223_005037/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v8_ablate_v7p1_R2_h15_mp001/20260223_005037/table2_kpis_raw.csv`

### R3（`adm_h=15` + 放宽进度阈值：`min_progress_m=0.0`）
- 命令：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models runs/v7p1_train300_esbest/train_20260221_010743 --out v8_ablate_v7p1_R3_h15_mp000 --runs 3 --forest-adm-horizon 15 --forest-min-progress-m 0.0`
- infer：
  - `run_dir`：`runs/v8_ablate_v7p1_R3_h15_mp000/20260223_005102`
  - `run_json`：`runs/v8_ablate_v7p1_R3_h15_mp000/20260223_005102/configs/run.json`
  - `kpi_mean_raw`：`runs/v8_ablate_v7p1_R3_h15_mp000/20260223_005102/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v8_ablate_v7p1_R3_h15_mp000/20260223_005102/table2_kpis_raw.csv`

### R4（允许轻微“负进度”：`min_progress_m=-0.02`）
- 命令：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models runs/v7p1_train300_esbest/train_20260221_010743 --out v8_ablate_v7p1_R4_h15_mpn002 --runs 3 --forest-adm-horizon 15 --forest-min-progress-m -0.02`
- infer：
  - `run_dir`：`runs/v8_ablate_v7p1_R4_h15_mpn002/20260223_005128`
  - `run_json`：`runs/v8_ablate_v7p1_R4_h15_mpn002/20260223_005128/configs/run.json`
  - `kpi_mean_raw`：`runs/v8_ablate_v7p1_R4_h15_mpn002/20260223_005128/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v8_ablate_v7p1_R4_h15_mpn002/20260223_005128/table2_kpis_raw.csv`

### R5（strict-argmax 诊断：`--forest-no-fallback`）
- 命令：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models runs/v7p1_train300_esbest/train_20260221_010743 --out v8_ablate_v7p1_R5_strict_argmax --runs 3 --forest-adm-horizon 30 --forest-min-progress-m 0.01 --forest-no-fallback`
- infer：
  - `run_dir`：`runs/v8_ablate_v7p1_R5_strict_argmax/20260223_005153`
  - `run_json`：`runs/v8_ablate_v7p1_R5_strict_argmax/20260223_005153/configs/run.json`
  - `kpi_mean_raw`：`runs/v8_ablate_v7p1_R5_strict_argmax/20260223_005153/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v8_ablate_v7p1_R5_strict_argmax/20260223_005153/table2_kpis_raw.csv`
