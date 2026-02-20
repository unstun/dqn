# v6p3 版本说明

- 版本类型：**Major（v+1）**
- 上一版本：`v6p2p5`
- 本版口径：`shielded/hybrid`（推理允许安全过滤与替换）
- 状态：**已完成 smoke（120ep + runs=3），待 full runs=20**

## 本版目标
- 将 `rl_algos`（训练/推理算法列表）收敛为仅 `cnn-ddqn`（卷积双 Q 网络）。
- 对比基线仅保留 `Hybrid A*-MPC`（混合 A* 全局规划 + MPC 局部跟踪）。
- 停止在本版本继续迭代 `DDPG/SAC`，避免研究主线分散。

## 方法概要
- 训练 profile：仅执行 `cnn-ddqn`。
- 推理 profile：仅评估 `cnn-ddqn`，并与 `Hybrid A*-MPC` 对照。
- 保持 `forest` 环境关键口径（reward、admissibility、top-k、stop override）与 `v6p2p5` 一致，确保可比性。

## 关键命令
- 自检：
  - `conda run -n ros2py310 python train.py --profile v6p3 --self-check`
  - `conda run -n ros2py310 python infer.py --profile v6p3 --self-check`
- 训练/推理：
  - `conda run -n ros2py310 python train.py --profile v6p3`
  - `conda run -n ros2py310 python infer.py --profile v6p3`

## 代表 run
- 训练：`runs/v6p3_smoke120/train_20260219_041557`
- 推理：`runs/v6p3_smoke120/train_20260219_041557/infer/20260219_043629`

## 当前结论
- smoke（runs=3）下，`CNN-DDQN` 在 `short/long` 达到 `SR=1.000`，但 `mid` 为 `SR=0.333`。
- `Hybrid A*-MPC` 在 `short/mid/long` 均为 `SR=1.000`。
- 当前主要问题集中在 `mid` 套件的碰撞失败（`collision`）。

## 下一步
1. 运行 smoke（建议：`--episodes 120` + `short/mid/long` 各 `runs=3`）。
2. smoke 通过后运行 full（建议：`--episodes 300` + `runs=20`）。
3. 在本版本四件套中回填 run 路径、KPI 与 failure_reason 分布。
