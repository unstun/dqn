# v7p3p2 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p3p1`（失败归档）
- 稳定对照基线：`v7p1`
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；主线保持 `v7p1`**

## 本版目标
- 针对 `v7p3p1` 的“遇障急拐导致绕路”问题，抑制推理期激进转向，同时允许短时小幅回退（绕障）以改善 `avg_path_length` 与 `path_time_s`。

## 方法摘要（多变量联动）
- 新增 turn-aware 替换重排：
  - `forest_topk_turn_penalty=1.0`
- 放松短视进度门：
  - `forest_min_progress_m=0.01 -> -0.01`
- 加强训练平滑惩罚：
  - `forest_reward_k_delta=0.8 -> 1.1`
- 保留并轻化自适应 no-progress 惩罚：
  - `forest_train_no_progress_penalty_dist_gain=0.15 -> 0.10`
  - `forest_train_no_progress_penalty_max=0.50 -> 0.45`

## 本轮关键命令（实际执行）
- 本地 -> 远端同步（不含 `runs/`）：
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

## 代表 run
- 训练：`runs/v7p3p2_turnaware_smoke/train_20260222_101744`
- 推理：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842`
- KPI（均值）：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.333 | 27.4510 | 19.2000 | 0.332357 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 33.7576 | 22.7500 | 0.246386 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 55.8795 | 33.8000 | 0.165122 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=1`，`timeout=2`
- mid：`reached=2`，`timeout=1`
- long：`reached=1`，`timeout=2`
- 合计：`reached=4`，`timeout=5`

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**。
- 相对 `v7p3p1`：
  - `avg_path_length/path_time_s` 在三套件有不同程度回落（long 回落最明显）；
  - 但 `success_rate` 显著下降（short `0.667 -> 0.333`、mid `1.000 -> 0.667`、long `1.000 -> 0.333`），且 short/mid 的曲率退化。
- 结论：当前改动降低了绕行代价，但把可达性压低到不可接受范围，不进入 full（`runs=20`），`v7p3p2` 失败归档。
- 下一版建议：`v7p3p3` 保留 turn-aware 结构，优先下调 `forest_topk_turn_penalty` 并按套件/距离自适应，避免过度抑制必要转向。
