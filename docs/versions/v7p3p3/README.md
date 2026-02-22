# v7p3p3 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p3p2`（失败归档）
- 稳定对照基线：`v7p1`
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；主线保持 `v7p1`**

## 本版目标
- 基于 `v7p3p2` 的推理侧消融结论，优先降低 `timeout`，尝试恢复 short/mid `success_rate`，同时不破坏 long 的可达性。

## 方法摘要（单变量族：推理门 + turn penalty）
- 下调 turn-aware 替换重排惩罚（训练/推理一致）：
  - `forest_topk_turn_penalty=1.0 -> 0.3`
- 收紧“允许负进度”的短视门（训练/推理一致）：
  - `forest_min_progress_m=-0.01 -> 0.0`
- 其余保持 `v7p3p2`：
  - 训练平滑惩罚：`forest_reward_k_delta=1.1`
  - 自适应 no-progress 惩罚：`dist_gain=0.10`，`max=0.45`

## 本轮关键命令（实际执行）
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260222_v7p3p3_infergate_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p3_infergate_smoke --models v7p3p3_infergate_smoke --out v7p3p3_infergate_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p3_infergate_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3p3_infergate_smoke/`

## 代表 run
- 训练：`runs/v7p3p3_infergate_smoke/train_20260222_112955`
- 推理：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657`
- KPI（均值）：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

> 说明：当 `success_rate=0` 时，`avg_path_length/path_time_s/avg_curvature_1_m` 为 `N/A`（该套件无到达样本）。

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.000 | N/A | N/A | N/A |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 26.4057 | 14.6000 | 0.128175 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.667 | 61.6431 | 32.4000 | 0.151448 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`collision=1`，`timeout=2`
- mid：`reached=1`，`collision=1`，`timeout=1`
- long：`reached=2`，`timeout=1`
- 合计：`reached=3`，`timeout=4`，`collision=2`

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**。
- 现象：
  - `argmax_inadmissible_rate` 相对 `v7p3p2` 有明显回落（替换触发更少），long `success_rate` 从 `0.333 -> 0.667`；
  - 但 short/mid 出现 `collision/timeout` 集中，short `success_rate` 下降到 `0.0`，整体不可接受。
- 处理：`v7p3p3` 失败归档，不进入 full（`runs=20`），主线保持 `v7p1`。
- 下一版建议：`v7p3p4` 优先做“推理期参数与训练期解耦”（仅推理侧应用 `tp=0.3/min_prog=0.0`，训练保持 `v7p3p2` 口径）或引入 `q_margin`（Q 差距阈值）避免过度平滑导致碰撞/超时。

