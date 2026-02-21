# v7p2p8 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p7`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；代码保持当前实现并继续前向迭代**

## 本版目标
- 在 `v7p2p7`（long 有恢复、short 退化）基础上做一次“大胆多变量”尝试，重点抑制 `timeout` 并提升 long 稳定性。
- 保持 `cnn-ddqn + globalcnn_fusion + foundation-fix` 主干不变，仅在训练稳定器上做三联改动。

## 方法摘要
- 多变量改动（相对 `v7p2p7`）：
  - 开启 `forest_train_dynamic_curriculum=true`（训练期 short/long 动态课程）。
  - 开启 `forest_expert_exploration=true`（训练期专家行为混入）。
  - 提高 `forest_reward_no_progress_penalty: 0.35 -> 0.45`（无进展惩罚增强）。
- 其余关键设置保持不变：
  - `grad_clip_norm=10.0`。
  - `reward_scale=0.1`、`reward_clip_abs=10.0`。
  - `forest_demo_pretrain_min_effective_steps=8000`。
- 复现配置：
  - `configs/v7p2p8.json`
  - `configs/repro_20260221_v7p2p8_bold_dynamic_expert_smoke.json`

## 联网调研记录（2026-02-21）
- 论文（近两年优先）：
  - Recent advances in reinforcement learning-based autonomous driving behavior planning: A survey（2024）
    - `https://doi.org/10.1016/j.trc.2024.104654`
    - 对应关系：支持“课程学习 + 统一评测口径”作为当前迭代主线。
  - Cross-Observability Optimistic-Pessimistic Safe Reinforcement Learning...（2024）
    - `https://doi.org/10.1109/tits.2024.3443397`
    - 对应关系：支持在不确定性/遮挡条件下加入稳健约束思路。
  - A Reinforcement Learning-Boosted Motion Planning Framework...（2024）
    - `https://doi.org/10.1109/iv55156.2024.10588750`
    - 对应关系：支持“RL+规划器协同”而非单一路径依赖。
  - CarPlanner: Consistent Auto-regressive Trajectory Planning...（2025）
    - `https://doi.org/10.1109/cvpr52734.2025.01607`
    - 对应关系：支持大规模训练一致性目标设计。
- 代码仓库（活跃/高 star）：
  - `https://github.com/DLR-RM/stable-baselines3`
  - `https://github.com/vwxyzjn/cleanrl`
  - `https://github.com/ray-project/ray`
  - `https://github.com/carla-simulator/carla`
  - `https://github.com/TUM-AVS/Frenetix-RL`

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p8_bold_dynamic_expert_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p8_bold_dynamic_expert_smoke --models v7p2p8_bold_dynamic_expert_smoke --out v7p2p8_bold_dynamic_expert_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p8_bold_dynamic_expert_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p8_bold_dynamic_expert_smoke/`

## 代表 run
- 训练：`runs/v7p2p8_bold_dynamic_expert_smoke/train_20260221_225358`
- 推理：`runs/v7p2p8_bold_dynamic_expert_smoke/train_20260221_225358/infer/20260221_230426`
- KPI（均值）：`runs/v7p2p8_bold_dynamic_expert_smoke/train_20260221_225358/infer/20260221_230426/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p8_bold_dynamic_expert_smoke/train_20260221_225358/infer/20260221_230426/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.000 | N/A | N/A | N/A |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 35.1857 | 30.9250 | 0.279375 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 1.000 | 57.0453 | 31.5667 | 0.266364 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`timeout=2`，`collision=1`
- mid：`reached=2`，`timeout=1`
- long：`reached=3`
- 合计：`reached=5`，`timeout=3`，`collision=1`

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**。
- 结果特征：
  - long 明显恢复（`SR: 0.333 -> 1.000`）；
  - short 显著崩塌（`SR: 0.333 -> 0.000`）；
  - mid/long 的 `path_time_s` 仍明显落后基线。
- 不满足 smoke 门“明确收益”条件，不进入 full（`runs=20`）。
- 下一版建议：`v7p2p9` 做分解消融（先保留动态课程，弱化专家混入或惩罚强度），避免 short/long 剧烈此消彼长。
