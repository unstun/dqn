# v7p2p10 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p9`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；代码保持当前实现并继续前向迭代**

## 本版目标
- 在 `v7p2p9`（short 回升但 long 崩塌）基础上继续单变量验证。
- 保持 `forest_expert_exploration=false` 与动态课程不变，仅降低 `no-progress` 惩罚强度，观察 short/long 跷跷板是否缓解。

## 方法摘要
- 单变量改动（相对 `v7p2p9`）：
  - `forest_reward_no_progress_penalty: 0.45 -> 0.35`
- 保持不变：
  - `forest_expert_exploration=false`
  - `forest_train_dynamic_curriculum=true`
  - `grad_clip_norm=10.0`
  - `reward_scale=0.1`、`reward_clip_abs=10.0`
  - `forest_demo_pretrain_min_effective_steps=8000`
- 复现配置：
  - `configs/v7p2p10.json`
  - `configs/repro_20260221_v7p2p10_penalty035_smoke.json`

## 联网调研说明（2026-02-21）
- 本轮为小改动（`px`）轻量调研，重点看“动态环境导航 RL”与“可复用工程实现”。
- 论文/综述（开放可访问）：
  - `Robots`: https://www.mdpi.com/2218-6581/14/8/95
  - `Sensors`: https://www.mdpi.com/1424-8220/25/15/4735
- GitHub 仓库（实时元数据）：
  - `tomasvr/turtlebot3_drlnav`（star=309，updated=2026-02-10）：https://github.com/tomasvr/turtlebot3_drlnav
  - `Arena-Rosnav/rosnav-rl`（star=73，updated=2026-02-05）：https://github.com/Arena-Rosnav/rosnav-rl
- 本轮不直接引入新模块：当前目标是隔离单变量贡献，先保持实验可归因。

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p10_penalty035_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p10_penalty035_smoke --models v7p2p10_penalty035_smoke --out v7p2p10_penalty035_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p10_penalty035_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p10_penalty035_smoke/`

## 代表 run
- 训练：`runs/v7p2p10_penalty035_smoke/train_20260221_234022`
- 推理：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340`
- KPI（均值）：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 26.7961 | 25.0500 | 0.269396 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 24.0931 | 13.3500 | 0.079707 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 51.0399 | 27.3500 | 0.164422 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=2`，`timeout=1`
- mid：`reached=1`，`timeout=2`
- long：`reached=1`，`timeout=1`，`collision=1`
- 合计：`reached=4`，`timeout=4`，`collision=1`

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**。
- 相比 `v7p2p9`：
  - long `SR` 从 `0.000 -> 0.333`（恢复）
  - mid 的 path/time 明显改善（`43.5747/23.2 -> 24.0931/13.35`）
  - short 的 path/smoothness 明显退化（`16.3453/0.118 -> 26.7961/0.269`）
- 结论：仍未形成“全套件明确收益”，不进入 full（`runs=20`）。
- 下一版建议：`v7p2p11` 保持其余参数不变，仅回调到中间值 `forest_reward_no_progress_penalty: 0.35 -> 0.40` 做单变量折中验证。
