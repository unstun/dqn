# v7p2p2 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p1`（失败归档）
- 回退基线：`v7p1`（当前稳定主线）
- 本版口径：`shielded/hybrid`（仅改网络模块，不改推理口径）
- 状态：**已运行（smoke：episodes=150, runs=3），失败归档；主线保持 `v7p1`**

## 本版目标
- 在保持 `cnn-ddqn`（Double DQN 目标计算）训练/推理流程不变的前提下，引入 `GlobalCNN`（全局多尺度卷积骨干）以提升中长程路径质量潜力。
- 将模块级改动收敛到最小范围，确保旧 checkpoint 兼容可加载。

## 方法摘要
- 网络改动：
  - `CNNQNetwork`（CNN Q 网络）新增 `cnn_backbone`（骨干选择）：`legacy` / `globalcnn`。
  - 新增 `globalcnn_width`（全局骨干基通道数）与 `globalcnn_dropout`（全局特征 dropout 概率）。
- Agent 改动：
  - `AgentConfig`（智能体配置）新增 GlobalCNN 参数并在 `DQNFamilyAgent`（DQN 系列智能体）中透传。
  - `save/load`（模型保存与加载）保持旧格式兼容：旧 checkpoint 无 GlobalCNN 字段时默认回退 `legacy`。
- 训练入口改动：
  - `train.py` CLI 新增 `--cnn-backbone`、`--cnn-global-width`、`--cnn-global-dropout`。
- 复现配置：
  - `configs/repro_20260221_v7p2p2_globalcnn_smoke.json`（本版 smoke 配置）。

## 本轮关键命令
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check（ubuntu-zt）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke 训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260221_v7p2p2_globalcnn_smoke"`
- 远端 smoke 推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260221_v7p2p2_globalcnn_smoke --models v7p2p2_globalcnn_smoke --out v7p2p2_globalcnn_smoke"`
- 远端 -> 本地回传结果：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2p2_globalcnn_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2p2_globalcnn_smoke/`
- 本地单测：
  - `conda run -n ros2py310 python -m pytest tests/test_globalcnn_network.py -v`

## 代表 run
- 训练：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611`
- 推理：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943`
- KPI（均值）：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 15.8975 | 10.1250 | 0.054351 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 24.1420 | 17.4500 | 0.110135 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 44.2185 | 28.3000 | 0.104250 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=2`，`timeout=1`
- mid：`reached=1`，`collision=1`，`timeout=1`
- long：`reached=1`，`timeout=2`
- 合计：`reached=4`，`timeout=4`，`collision=1`

## 结论与下一步
- 本轮 smoke 未通过：虽然 short 的路径长度与时间有局部收益，但 `success_rate` 在 short/mid/long 均显著低于 `Hybrid A*-MPC`，mid/long 的时间与平滑性也无优势。
- 按版本工作流规则，`v7p2p2` 记为失败归档，不进入 full（`runs=20`）评测，主线保持 `v7p1`。
- 下一轮建议以 `v7p2p3` 命名继续模块迭代，并优先提升中长程可达率与超时控制。
