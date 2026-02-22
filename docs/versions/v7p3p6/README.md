# v7p3p6 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p3p4`（失败归档）
- 稳定对照基线：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**失败归档（smoke NO-GO）**

## 本版目标
- 在保持 `cnn-ddqn` 与 `globalcnn_fusion`（全局卷积融合主干）不变的前提下，继续使用 `obs_map_size=128`（占据图分辨率）并针对 long 套件超时问题做参数纠偏。

## 方法摘要
- 算法与模块不变：
  - 算法：`cnn-ddqn`
  - 主干：`globalcnn_fusion`
  - 推理口径：`forest_no_fallback=false`
- 本版参数主改动：
  - `obs_map_size=128`（train/infer 同步）
  - `replay_capacity=8000`、`batch_size=32`（资源保护）
  - `forest_topk_turn_penalty: 1.0 -> 0.3`（减弱替换动作转向惩罚）
  - `forest_min_progress_m: -0.01 -> 0.0`（收紧进展判定）
  - `forest_train_short_prob: 0.35 -> 0.25`（训练采样偏向 long）
  - `forest_train_dynamic_target_sr_long: 0.75 -> 0.85`（提高 long 目标）
  - demo 预算调整：`forest_demo_target_cap=8000`、`forest_demo_pretrain_steps=12000`

## 本轮关键命令（计划/执行）
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260222_v7p3p6_obsmap128_tune_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p6_obsmap128_tune_smoke --models runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007"`

## 代表 run
- 训练：`runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007`
- 推理：`runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007/infer/20260222_223831`
- KPI（均值）：`runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007/infer/20260222_223831/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007/infer/20260222_223831/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s |
|---|---|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 24.1966 | 13.5000 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 |
| mid | CNN-DDQN | 0.333 | 29.0903 | 16.9500 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 |
| long | CNN-DDQN | 0.333 | 67.9985 | 39.5500 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 |

与 `v7p3p5` 相比（同为 `obs_map_size=128`）：
- long 套件 `success_rate: 0.000 -> 0.333`（从 3/3 timeout 改善为 1/3 reached）。
- mid 套件 `avg_path_length/path_time_s: 69.6498/39.8500 -> 29.0903/16.9500`（显著下降）。
- short 套件 `path_time_s` 降低（`28.2 -> 13.5`），但 `avg_path_length` 略增（`23.0936 -> 24.1966`）。

## 结论与下一步
- 当前结论：
  - 工程可行：现有 CNN 在 `obs_map_size=128` 下可稳定完成 train+infer（无需换模块）。
  - 指标未通过：short/long 相对 `Hybrid A*-MPC` 三条门槛均未满足，smoke 判定 `NO-GO`。
  - 方向有效：long 从 `v7p3p5` 的 `0.000` 恢复到 `0.333`，说明参数纠偏方向可继续。
- 下一步：
  - 下一轮继续以 `v7p3p7` 前向迭代，优先降低 long 的 timeout（`2/3`）并压缩 long 路径与时间；
  - 稳定对照仍使用 `v7p1`，不回退当前代码线。
