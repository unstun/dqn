# v7p2p4 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p3`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（仅改网络模块，不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；代码保持当前实现并继续前向迭代**

## 本版目标
- 在保持 `cnn-ddqn`（Double DQN 目标计算）训练/推理流程不变前提下，给 `globalcnn_fusion` 增加空间先验（agent/goal heatmap）通道，提升中长程可达率与超时稳定性。

## 方法摘要
- 网络改动：
  - `CNNQNetwork` 新增 `globalcnn_spatial_prior`（是否启用空间先验）和 `globalcnn_prior_sigma`（先验高斯核宽度）。
  - 当 `cnn_backbone=globalcnn/globalcnn_fusion` 且开关开启时，将 `ax/ay/gx/gy`（车与目标归一化位置）转换为 2 个热力图通道并与占据图拼接后再卷积。
- Agent/CLI 改动：
  - `AgentConfig` 增加对应参数并贯通保存/加载。
  - `train.py` 新增 `--cnn-global-spatial-prior`、`--cnn-global-prior-sigma`。
- 测试改动：
  - `tests/test_globalcnn_network.py` 新增空间先验通道接线测试，roundtrip 覆盖新参数。
- 复现配置：
  - `configs/v7p2p4.json`
  - `configs/repro_20260221_v7p2p4_globalcnn_spatialprior_smoke.json`

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check（ubuntu-zt）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p4_globalcnn_spatialprior_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p4_globalcnn_spatialprior_smoke --models v7p2p4_globalcnn_spatialprior_smoke --out v7p2p4_globalcnn_spatialprior_smoke"`
- 远端 -> 本地回传结果：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p4_globalcnn_spatialprior_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p4_globalcnn_spatialprior_smoke/`

## 代表 run
- 训练：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908`
- 推理：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926`
- KPI（均值）：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 17.9810 | 10.7250 | 0.142219 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 28.8075 | 16.5500 | 0.323404 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 1.000 | 51.1817 | 28.2000 | 0.205800 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=2`，`collision=1`
- mid：`reached=2`，`timeout=1`
- long：`reached=3`
- 合计：`reached=7`，`timeout=1`，`collision=1`

## 结论与下一步
- 本轮 smoke 仍未通过最终门槛：short/mid 在 `success_rate` 仍落后，mid/long 的路径长度与时间仍显著劣于 `Hybrid A*-MPC`。
- 相对 `v7p2p3`，本轮在 short/mid/long `success_rate` 全部提升（`0.333/0.333/0.667 -> 0.667/0.667/1.000`），且 mid 的异常长轨迹显著缓解。
- 按你最新策略，本轮作为失败归档保留证据，**不回退代码**，继续在当前实现上进入下一轮（建议 `v7p2p5`）迭代。
