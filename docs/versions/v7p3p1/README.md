# v7p3p1 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p3`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；主线保持 `v7p1`**

## 本版目标
- 在 `v7p3`（short/mid 局部改善但 long path/time 退化）基础上，移除 short/long 套件硬编码惩罚，验证“按起终点距离比例自适应 no-progress 惩罚”是否能提升泛化并改善 long 套件。

## 方法摘要
- 单变量族改动（相对 `v7p3`）：
  - 关闭套件惩罚：`forest_train_suite_no_progress_penalty=false`
  - 开启通用自适应惩罚：`forest_train_adaptive_no_progress_penalty=true`
  - 自适应公式：`penalty = clip(base + gain * dist_ratio, min, max)`
    - `base=0.35`
    - `gain=0.15`
    - `min=0.35`
    - `max=0.50`
- 保持不变：
  - `forest_expert_exploration=false`
  - `forest_train_dynamic_curriculum=true`
  - `grad_clip_norm=10.0`
  - `reward_scale=0.1`、`reward_clip_abs=10.0`
  - `forest_no_fallback=false`（`shielded/hybrid`）
- 复现配置：
  - `configs/v7p3p1.json`
  - `configs/repro_20260222_v7p3p1_adaptive_penalty_smoke.json`

## 本轮关键命令（实际执行）
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260222_v7p3p1_adaptive_penalty_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p1_adaptive_penalty_smoke --models v7p3p1_adaptive_penalty_smoke --out v7p3p1_adaptive_penalty_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p1_adaptive_penalty_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3p1_adaptive_penalty_smoke/`

## 代表 run
- 训练：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303`
- 推理：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552`
- KPI（均值）：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 32.0322 | 24.3750 | 0.302475 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 1.000 | 34.4515 | 22.7667 | 0.189560 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 1.000 | 83.8999 | 48.1000 | 0.218480 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=2`，`timeout=1`
- mid：`reached=3`
- long：`reached=3`
- 合计：`reached=8`，`timeout=1`

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**。
- 相对 `v7p3`：
  - short：`SR` 持平（`0.667 -> 0.667`），但 `path/time/smoothness` 明显退化；
  - mid：`SR` 提升（`0.667 -> 1.000`），但 `path/time/smoothness` 退化；
  - long：`SR` 提升（`0.333 -> 1.000`），但 `path/time/smoothness` 显著退化。
- 结论：虽然成功率提升，但未满足当前研究目标（路径更短、时间更短、轨迹更平滑），不进入 full（`runs=20`），`v7p3p1` 失败归档。
- 下一版建议：`v7p3p2` 继续单变量，优先压缩 long 的 `avg_path_length/path_time_s`，并抑制 `argmax_inadmissible_rate` 上升。
