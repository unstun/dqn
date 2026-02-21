# v7p2p7 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p6`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；代码保持当前实现并继续前向迭代**

## 本版目标
- 聚焦 `v7p2p6` 的 long 崩塌（`SR=0.0`）做单变量恢复。
- 保持 `cnn-ddqn + globalcnn_fusion + foundation-fix` 其余设置不变，仅恢复梯度裁剪阈值：
  - `grad_clip_norm: 5.0 -> 10.0`

## 方法摘要
- 继承 `v7p2p6` 三项基础设施：
  - `reward_scale/reward_clip_abs`
  - `grad_norm_ema/grad_clip_hit_rate` 观测
  - `forest_demo_pretrain_min_effective_steps`
- 本版唯一变量：放宽梯度裁剪上限，验证是否能恢复 long 可达率。
- 复现配置：
  - `configs/v7p2p7.json`
  - `configs/repro_20260221_v7p2p7_gradclip_recover_smoke.json`

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p7_gradclip_recover_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p7_gradclip_recover_smoke --models v7p2p7_gradclip_recover_smoke --out v7p2p7_gradclip_recover_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p7_gradclip_recover_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p7_gradclip_recover_smoke/`

## 代表 run
- 训练：`runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452`
- 推理：`runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452/infer/20260221_221008`
- KPI（均值）：`runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452/infer/20260221_221008/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452/infer/20260221_221008/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.333 | 24.6699 | 14.0000 | 0.267312 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 36.4765 | 24.5500 | 0.234826 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 61.7273 | 32.2000 | 0.154668 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=1`，`timeout=2`
- mid：`reached=2`，`timeout=1`
- long：`reached=1`，`timeout=1`，`collision=1`
- 合计：`reached=4`，`timeout=4`，`collision=1`

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**。
- 相比 `v7p2p6`：
  - long 从 `SR=0.0` 恢复到 `0.333`（有恢复但不足）；
  - short 从 `SR=1.0` 回落到 `0.333`（明显退化）。
- 仍不满足 smoke 门“明确收益”条件，不进入 full（`runs=20`）。
- 下一版建议：`v7p2p8` 继续单变量，优先抑制 short 超时并稳定 long 可达率。
