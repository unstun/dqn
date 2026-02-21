# v7p2p3 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p2`（失败归档）
- 回退基线：`v7p1`（当前稳定主线）
- 本版口径：`shielded/hybrid`（仅改网络模块，不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；主线保持 `v7p1`**

## 本版目标
- 在保持 `cnn-ddqn`（Double DQN 目标计算）训练/推理流程不变前提下，对 `GlobalCNN` 增加“全局+局部双分支融合”，尝试修复 `v7p2p2` 的中长程超时问题。

## 方法摘要
- 网络改动：
  - `CNNQNetwork`（CNN Q 网络）新增 `cnn_backbone=globalcnn_fusion`（全局多尺度分支 + 局部分支 + 门控融合）。
  - 保留原 `legacy/globalcnn`，保证可回退对照。
- 训练入口改动：
  - `train.py` 的 `--cnn-backbone` 增加 `globalcnn_fusion` 选项。
- 测试改动：
  - 扩展 `tests/test_globalcnn_network.py`，覆盖 `globalcnn_fusion` 前向维度与 checkpoint roundtrip。
- 复现配置：
  - `configs/repro_20260221_v7p2p3_globalcnn_fusion_smoke.json`（本版 smoke 配置）。

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check（ubuntu-zt）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p3_globalcnn_fusion_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p3_globalcnn_fusion_smoke --models v7p2p3_globalcnn_fusion_smoke --out v7p2p3_globalcnn_fusion_smoke"`
- 远端 -> 本地回传结果：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p3_globalcnn_fusion_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p3_globalcnn_fusion_smoke/`
- 本地单测：
  - `conda run -n ros2py310 python -m pytest tests/test_globalcnn_network.py -v`

## 代表 run
- 训练：`runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334`
- 推理：`runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334/infer/20260221_180256`
- KPI（均值）：`runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334/infer/20260221_180256/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334/infer/20260221_180256/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.333 | 13.1250 | 8.2000 | 0.005843 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 64.7312 | 48.2000 | 0.202941 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.667 | 48.5295 | 30.9000 | 0.169235 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=1`，`timeout=1`，`collision=1`
- mid：`reached=1`，`timeout=2`
- long：`reached=2`，`timeout=1`
- 合计：`reached=4`，`timeout=4`，`collision=1`

## 结论与下一步
- 本轮 smoke 未通过：long 相比 `v7p2p2` 有局部改善，但 short `success_rate` 进一步下降，mid 指标明显恶化（路径长度/时间显著变差）。
- 按版本工作流规则，`v7p2p3` 记为失败归档，不进入 full（`runs=20`）评测，主线保持 `v7p1`。
- 下一轮建议以 `v7p2p4` 延续模块迭代，优先约束 mid 套件的超时与异常长轨迹。
