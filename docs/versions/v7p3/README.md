# v7p3 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p10`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；主线保持 `v7p1`**

## 本版目标
- 在 `v7p2p10`（long 回升但 short 路径/平滑性退化）基础上，验证“训练期 short/long 分离 no-progress 惩罚”是否能缓解 short/long 跷跷板。

## 方法摘要
- 单变量族改动（相对 `v7p2p10`）：
  - `forest_train_suite_no_progress_penalty=true`
  - `forest_train_short_no_progress_penalty=0.45`
  - `forest_train_long_no_progress_penalty=0.35`
  - `forest_reward_no_progress_penalty=0.40`（基础回退值）
- 保持不变：
  - `forest_expert_exploration=false`
  - `forest_train_dynamic_curriculum=true`
  - `grad_clip_norm=10.0`
  - `reward_scale=0.1`、`reward_clip_abs=10.0`
  - `forest_no_fallback=false`（`shielded/hybrid`）
- 复现配置：
  - `configs/v7p3.json`
  - `configs/repro_20260221_v7p3_suite_penalty_smoke.json`

## 本轮关键命令（实际执行）
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p3_suite_penalty_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p3_suite_penalty_smoke --models v7p3_suite_penalty_smoke --out v7p3_suite_penalty_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3_suite_penalty_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3_suite_penalty_smoke/`

## 代表 run
- 训练：`runs/v7p3_suite_penalty_smoke/train_20260222_012415`
- 推理：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023`
- KPI（均值）：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 25.3610 | 17.4500 | 0.193361 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 34.0156 | 18.3250 | 0.148345 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 57.1796 | 30.1500 | 0.102428 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=2`，`timeout=1`
- mid：`reached=2`，`collision=1`
- long：`reached=1`，`timeout=1`，`collision=1`
- 合计：`reached=5`，`timeout=2`，`collision=2`

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**。
- 相对 `v7p2p10`：
  - short：`SR` 持平（`0.667`），`path/time/smoothness` 有改善；
  - mid：`SR` 提升（`0.333 -> 0.667`），但 `path/time/smoothness` 退化；
  - long：`SR` 持平（`0.333`），`path/time` 退化。
- 结论：未形成“全套件明确收益”，不进入 full（`runs=20`），`v7p3` 失败归档。
- 下一版建议：`v7p3p1` 继续单变量，优先修复 long 套件 path/time 退化。
