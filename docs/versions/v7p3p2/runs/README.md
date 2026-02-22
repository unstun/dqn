# v7p3p2 runs 追溯

## 1) 本轮执行命令（实际）
- 本地 -> 远端同步：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260222_v7p3p2_turnaware_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p2_turnaware_smoke --models v7p3p2_turnaware_smoke --out v7p3p2_turnaware_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p2_turnaware_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3p2_turnaware_smoke/`

## 2) run 路径登记
- train：
  - `run_dir`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744`
  - `run_json`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/configs/run.json`
  - `train_meta`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842`
  - `run_json`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842/table2_kpis_raw.csv`

## 3) 关键参数快照
- 复现配置：`configs/repro_20260222_v7p3p2_turnaware_smoke.json`
- 相对 `v7p3p1` 的关键差异：
  - `train.forest_topk_turn_penalty=1.0`
  - `infer.forest_topk_turn_penalty=1.0`
  - `train.forest_min_progress_m=-0.01`
  - `infer.forest_min_progress_m=-0.01`
  - `train.forest_reward_k_delta=1.1`
  - `train.forest_train_no_progress_penalty_dist_gain=0.10`
  - `train.forest_train_no_progress_penalty_max=0.45`
- 训练元信息（`train_meta_forest_a.json` / `algos.cnn-ddqn`）：
  - `forest_topk_turn_penalty=1.0`
  - `adaptive_no_progress_penalty_enabled=true`
  - `adaptive_no_progress_dist_gain=0.1`
  - `active_no_progress_penalty_mean=0.384496622578361`
  - `active_no_progress_penalty_min=0.36225943235914504`
  - `active_no_progress_penalty_max=0.4200469877223725`
  - `adaptive_dist_ratio_mean=0.34496622578361047`

## 4) short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=0.333`，`avg_path_length=27.4510`，`path_time_s=19.2000`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- long：
  - CNN-DDQN：`success_rate=0.333`，`avg_path_length=55.8795`，`path_time_s=33.8000`
  - Hybrid A*-MPC：`success_rate=1.000`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 5) `failure_reason` 分布
- CNN-DDQN：
  - short：`reached=1`，`timeout=2`
  - mid：`reached=2`，`timeout=1`
  - long：`reached=1`，`timeout=2`
  - 合计：`reached=4`，`timeout=5`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## 6) 推理侧快速消融（固定模型，仅改 `forest_topk_turn_penalty` / `forest_min_progress_m`）

### 6.1 命令（实际）
- 固定模型来源（不重训）：
  - `models_run_dir`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744`
- 远端推理（逐组覆盖参数，runs=3，seed=33）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p2_turnaware_smoke --models runs/v7p3p2_turnaware_smoke/train_20260222_101744 --out <OUT> --seed 33 --runs 3 --forest-topk-turn-penalty <TP> --forest-min-progress-m <MP>"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p2_turnaware_smoke_ablate_* /home/sun/phdproject/dqn/dqn/runs/`

### 6.2 结果摘要（CNN-DDQN，runs=3）

> 记号：`SR (r/t/c)` 表示 `success_rate` 以及 `failure_reason` 的 `reached/timeout/collision` 计数。

| tp | min_prog | short SR (r/t/c) | mid SR (r/t/c) | long SR (r/t/c) | run_dir |
|---:|---:|---|---|---|---|
| 0.0 | -0.01 | 0.000 (r=0, t=3, c=0) | 0.333 (r=1, t=1, c=1) | 1.000 (r=3, t=0, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p0_mpm0p01/20260222_111111` |
| 0.0 | 0.00 | 0.333 (r=1, t=2, c=0) | 0.667 (r=2, t=1, c=0) | 1.000 (r=3, t=0, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p0_mp0p0/20260222_111135` |
| 0.0 | 0.01 | 0.333 (r=1, t=2, c=0) | 0.667 (r=2, t=1, c=0) | 0.667 (r=2, t=1, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p0_mp0p01/20260222_111021` |
| 0.3 | -0.01 | 0.667 (r=2, t=1, c=0) | 0.667 (r=2, t=1, c=0) | 0.333 (r=1, t=2, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p3_mpm0p01/20260222_111202` |
| 0.3 | 0.00 | 0.667 (r=2, t=1, c=0) | 0.667 (r=2, t=1, c=0) | 1.000 (r=3, t=0, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p3_mp0p0/20260222_111227` |
| 0.3 | 0.01 | 0.667 (r=2, t=1, c=0) | 1.000 (r=3, t=0, c=0) | 0.667 (r=2, t=1, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p3_mp0p01/20260222_111250` |
| 0.6 | -0.01 | 0.333 (r=1, t=2, c=0) | 0.667 (r=2, t=1, c=0) | 1.000 (r=3, t=0, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p6_mpm0p01/20260222_111313` |
| 0.6 | 0.00 | 0.333 (r=1, t=2, c=0) | 0.667 (r=2, t=1, c=0) | 1.000 (r=3, t=0, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p6_mp0p0/20260222_111343` |
| 0.6 | 0.01 | 0.333 (r=1, t=2, c=0) | 0.667 (r=2, t=0, c=1) | 0.667 (r=2, t=1, c=0) | `runs/v7p3p2_turnaware_smoke_ablate_tp0p6_mp0p01/20260222_111410` |

### 6.3 工件路径约定（每行通用）
- `run_json`：`<run_dir>/configs/run.json`
- `kpi_mean_raw`：`<run_dir>/table2_kpis_mean_raw.csv`
- `kpi_raw`：`<run_dir>/table2_kpis_raw.csv`

### 6.4 初步结论（仅 smoke 门参考）
- `forest_topk_turn_penalty=1.0`（v7p3p2 原配置）对 long/short 的 `timeout` 过于敏感。
- 在固定模型前提下，`tp=0.3` 且 `min_prog=0.0` 给出更均衡的 SR（short/mid/long=0.667/0.667/1.000）；可作为 `v7p3p3` 的首选推理侧参数候选。
