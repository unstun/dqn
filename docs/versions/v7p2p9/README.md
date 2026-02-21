# v7p2p9 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p8`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；代码保持当前实现并继续前向迭代**

## 本版目标
- 在 `v7p2p8`（long 恢复、short 崩塌）基础上做分解消融。
- 保持动态课程与惩罚强度不变，仅关闭训练期专家混入，验证 short/long 波动是否由 `expert exploration` 主导。

## 方法摘要
- 单变量改动（相对 `v7p2p8`）：
  - `forest_expert_exploration: true -> false`
- 保持不变：
  - `forest_train_dynamic_curriculum=true`
  - `forest_reward_no_progress_penalty=0.45`
  - `grad_clip_norm=10.0`
  - `reward_scale=0.1`、`reward_clip_abs=10.0`
  - `forest_demo_pretrain_min_effective_steps=8000`
- 复现配置：
  - `configs/v7p2p9.json`
  - `configs/repro_20260221_v7p2p9_ablate_expert_smoke.json`

## 联网调研说明
- 本轮属于小改动单变量消融，沿用 `v7p2p8` 已完成的联网调研结论，不新增检索。
- 跳过理由：当前目标是隔离单一训练变量影响，仓库内证据（`v7p2p8` run + train-progress）已足够支撑本轮实验设计。

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p9_ablate_expert_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p9_ablate_expert_smoke --models v7p2p9_ablate_expert_smoke --out v7p2p9_ablate_expert_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p9_ablate_expert_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p9_ablate_expert_smoke/`

## 代表 run
- 训练：`runs/v7p2p9_ablate_expert_smoke/train_20260221_231402`
- 推理：`runs/v7p2p9_ablate_expert_smoke/train_20260221_231402/infer/20260221_232825`
- KPI（均值）：`runs/v7p2p9_ablate_expert_smoke/train_20260221_231402/infer/20260221_232825/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p9_ablate_expert_smoke/train_20260221_231402/infer/20260221_232825/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 16.3453 | 26.2750 | 0.118230 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 43.5747 | 23.2000 | 0.167997 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.000 | N/A | N/A | N/A |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=2`，`timeout=1`
- mid：`reached=1`，`timeout=1`，`collision=1`
- long：`timeout=3`
- 合计：`reached=3`，`timeout=5`，`collision=1`

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**。
- 相比 `v7p2p8`：
  - short 从 `SR=0.0` 回升到 `0.667`；
  - long 从 `SR=1.0` 回落到 `0.0`。
- 说明当前系统仍存在显著 short/long 跷跷板效应，不满足“明确收益”门槛，不进入 full（`runs=20`）。
- 下一版建议：`v7p2p10` 在 `v7p2p9` 基础上保持 `expert_exploration=false`，仅回退 `forest_reward_no_progress_penalty: 0.45 -> 0.35` 做单变量验证。
