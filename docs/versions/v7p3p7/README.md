# v7p3p7 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p3p6`（失败归档）
- 稳定对照基线：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**失败归档（smoke NO-GO）**

## 本版目标
- 在保持 `cnn-ddqn` 与 `globalcnn_fusion`（全局卷积融合主干）不变的前提下，继续使用 `obs_map_size=128`（占据图分辨率），优先降低 long 套件 `timeout`（超时失败）并压缩 long 路径与时间。

## 方法摘要
- 算法与模块不变：
  - 算法：`cnn-ddqn`
  - 主干：`globalcnn_fusion`
  - 推理口径：`forest_no_fallback=false`
- 本版参数主改动（相对 `v7p3p6`）：
  - `forest_topk: 10 -> 12`
  - `forest_topk_turn_penalty: 0.3 -> 0.2`
  - `forest_min_progress_m: 0.0 -> 0.02`
  - `forest_train_short_prob: 0.25 -> 0.20`
  - `forest_train_dynamic_target_sr_long: 0.85 -> 0.90`
  - `forest_train_dynamic_min_short_prob: 0.25 -> 0.20`
  - `forest_demo_pretrain_steps: 12000 -> 15000`
  - `forest_demo_pretrain_min_effective_steps: 3000 -> 4000`
  - `forest_demo_pretrain_val_runs: 6 -> 8`
  - `replay_capacity: 8000 -> 10000`

## 本轮关键命令（执行）
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke --models runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248"`

## 代表 run
- 训练：`runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248`
- 推理：`runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329`
- KPI（均值）：`runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s |
|---|---|---:|---:|---:|
| short | CNN-DDQN | 1.000 | 21.3138 | 13.9000 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 |
| mid | CNN-DDQN | 1.000 | 36.6758 | 21.2500 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 |
| long | CNN-DDQN | 0.333 | 66.8440 | 35.7500 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 |

与 `v7p3p6` 相比（同为 `obs_map_size=128`）：
- short：`SR 0.667 -> 1.000`，`avg_path_length 24.1966 -> 21.3138`，`path_time_s 13.5 -> 13.9`（略升）。
- mid：`SR 0.333 -> 1.000`，但 `avg_path_length/path_time_s` 从 `29.0903/16.9500` 退化到 `36.6758/21.2500`。
- long：`SR` 持平 `0.333`，`avg_path_length/path_time_s` 从 `67.9985/39.5500` 降到 `66.8440/35.7500`（小幅改善）。
- `failure_reason` 总体：CNN 从 `reached=4, timeout=5` 改善为 `reached=7, timeout=2`，但 long 仍是 `1 reached + 2 timeout`。

## 结论与下一步
- 当前结论：
  - 工程可行：现有 CNN 在 `obs_map_size=128` 下可稳定完成 train+infer（无需换模块）。
  - 指标未通过：short/long 相对 `Hybrid A*-MPC` 三条门槛不等式仍未同时满足，smoke 判定 `NO-GO`。
  - 方向有效：超时总量从 5 降到 2，short/mid SR 提升到 1.0，但 long 可达性仍未突破。
- 下一步：
  - 进入 `v7p3p8`，优先针对 long `2/3 timeout` 做定向优化（保持当前模块，不回退代码线）。
