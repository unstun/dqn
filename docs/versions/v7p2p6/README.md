# v7p2p6 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p5`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；代码保持当前实现并继续前向迭代**

## 本版目标
- 先修训练基础设施三项：
  - `reward scaling/clipping`（奖励缩放/裁剪）
  - `gradient clipping observability`（梯度裁剪可观测）
  - `effective pretrain gate`（预训练最小有效步数门控）
- 保持 `cnn-ddqn + globalcnn_fusion` 主干不变，隔离变量。

## 方法摘要
- 奖励变换：在 `observe(...)` 入 replay 前统一执行 `reward' = clip(reward * reward_scale)`。
- 梯度观测：在 `update()`/`pretrain_on_demos()` 记录并回传 `grad_norm_pre_clip` 与 `grad_clip_hit`。
- 预训练门控：早停需满足 `min_effective_steps`，防止“只跑很少更新就停”。
- 复现配置：
  - `configs/v7p2p6.json`
  - `configs/repro_20260221_v7p2p6_foundationfix_smoke.json`

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p6_foundationfix_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p6_foundationfix_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p6_foundationfix_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p6_foundationfix_smoke/`

## 代表 run
- 训练：`runs/v7p2p6_foundationfix_smoke/train_20260221_211603`
- 推理：`runs/v7p2p6_foundationfix_smoke/train_20260221_211603/infer/20260221_213248`
- KPI（均值）：`runs/v7p2p6_foundationfix_smoke/train_20260221_211603/infer/20260221_213248/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p6_foundationfix_smoke/train_20260221_211603/infer/20260221_213248/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 1.000 | 20.5613 | 18.2667 | 0.166511 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 34.4835 | 36.6500 | 0.307094 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.000 | N/A | N/A | N/A |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=3`
- mid：`reached=2`，`timeout=1`
- long：`timeout=3`
- 合计：`reached=5`，`timeout=4`

## 结论与下一步
- 本轮 smoke 仍为 **NO-GO**：
  - short 有改善，但仍在 `avg_path_length/path_time_s` 落后 Hybrid；
  - mid/long 未过门，特别是 long `SR=0.0`。
- 但三项基础设施已确认落地并生效（日志可追溯）：
  - `reward_scale/reward_clip_abs`
  - `grad_norm_ema/grad_clip_hit_rate`
  - `min_effective_steps` 门控早停
- 下一版 `v7p2p7` 建议仅做“long 失败恢复”单一改动，不回退代码。
