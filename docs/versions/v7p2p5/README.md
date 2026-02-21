# v7p2p5 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p4`（失败归档）
- 稳定对照基线：`v7p1`（当前对照口径）
- 本版口径：`shielded/hybrid`（仅改网络模块，不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；代码保持当前实现并继续前向迭代**

## 本版目标
- 取消“目标走廊先验”方案，仅在现有 `globalcnn_fusion + spatial_prior` 基础上加入融合归一化（LayerNorm），希望提升 mid/long 稳定性。

## 方法摘要
- 网络改动：
  - `CNNQNetwork` 新增 `globalcnn_fusion_layernorm`（融合归一化开关）与 `globalcnn_fusion_layernorm_eps`（LayerNorm epsilon）。
  - 在 `globalcnn_fusion` 的 `global_feat + local_feat` 融合后、门控前执行 `LayerNorm`（可开关）。
- Agent/CLI 改动：
  - `AgentConfig` 增加以上参数并在 `DQNFamilyAgent` 贯通到网络。
  - `train.py` 新增 `--cnn-fusion-layernorm`、`--cnn-fusion-layernorm-eps`。
- 测试改动：
  - `tests/test_globalcnn_network.py` 扩展 roundtrip 与结构接线，覆盖新参数。
- 复现配置：
  - `configs/v7p2p5.json`
  - `configs/repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke.json`

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check（ubuntu-zt）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke --models v7p2p5_globalcnn_fusionnorm_smoke --out v7p2p5_globalcnn_fusionnorm_smoke"`
- 远端 -> 本地回传结果：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p5_globalcnn_fusionnorm_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p5_globalcnn_fusionnorm_smoke/`

## 代表 run
- 训练：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023`
- 推理：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626`
- KPI（均值）：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.333 | 31.5744 | 17.1500 | 0.191482 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 42.0740 | 28.1500 | 0.128448 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.667 | 89.4701 | 51.0000 | 0.193102 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=1`，`timeout=2`
- mid：`reached=2`，`timeout=1`
- long：`reached=2`，`timeout=1`
- 合计：`reached=5`，`timeout=4`

## 结论与下一步
- 本轮 smoke 未通过，且相对 `v7p2p4` 全面退化（short/mid/long 的路径长度与时间显著变差，short SR 下滑到 0.333）。
- 说明当前“fusion LayerNorm 位置/配置”对本任务分布不适配，需要下一版改为更保守的融合稳定策略（例如仅门控分支归一化或降低融合强度）。
- 按你当前策略：失败版本保留归档证据，**不回退代码**，继续在当前实现上进入 `v7p2p6`。
