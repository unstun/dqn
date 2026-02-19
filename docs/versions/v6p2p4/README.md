# v6p2p4 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v6p2p3`
- 本版口径：`shielded/hybrid`（训练与推理统一，继承 v6p2p3）
- 状态：**已运行（训练 300 轮 + short/mid/long 各 5 轮推理）**

## 本版目标
- 在 `v6p2p3` 统一口径基础上，增加 `DDPG` 与 `SAC` 作为对比算法。
- 训练与推理统一评估 `cnn-ddqn / ddpg / sac`。
- 保持 `k_t=0.10`、`k_delta=0.8` 与统一策略规则不变。

## 方法概要
- 代码侧：
  - 新增连续控制算法实现：`DDPG`、`SAC`。
  - `train.py` 增加连续算法训练分支（`env.step_continuous(...)`）。
  - `infer.py` 增加连续算法推理 rollout 分支。
  - `benchmark.py` 扩展 `ddpg/sac` 算法白名单与标签。
- 配置侧：
  - 新增 `configs/v6p2p4.json`。

## 本轮执行记录（2026-02-19）
- 训练：`conda run -n ros2py310 python train.py --profile v6p2p4`
- 推理：`conda run -n ros2py310 python infer.py --profile v6p2p4`

## 关键命令
- 自检：
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 训练/推理：
  - `conda run -n ros2py310 python train.py --profile v6p2p4`
  - `conda run -n ros2py310 python infer.py --profile v6p2p4`

## 代表 run
- 训练：`runs/v6p2p4/train_20260219_153029`
- 推理：`runs/v6p2p4/train_20260219_153029/infer/20260219_161252`

## 本轮结果摘要（runs=5）
- short：
  - CNN-DDQN：`success_rate=0.80`，`avg_path_length=18.1736`，`path_time_s=19.00`
  - DDPG：`success_rate=0.00`（`collision=4`, `timeout=1`）
  - SAC：`success_rate=0.00`（`collision=4`, `timeout=1`）
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=16.8724`，`path_time_s=10.00`
- mid：
  - CNN-DDQN：`success_rate=0.80`，`avg_path_length=31.4601`，`path_time_s=19.45`
  - DDPG：`success_rate=0.00`（`collision=3`, `timeout=2`）
  - SAC：`success_rate=0.00`（`collision=4`, `timeout=1`）
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=25.1525`，`path_time_s=13.87`
- long：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=51.9006`，`path_time_s=30.42`
  - DDPG：`success_rate=0.00`（`collision=5`）
  - SAC：`success_rate=0.00`（`collision=5`）
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=43.0247`，`path_time_s=22.82`

## 下一步
1. 对连续算法做训练策略修订（如奖励重标定、动作约束 shaping、warm-start）后再复测。
2. 保持 `v6p2p3` 统一规则口径，继续 short/long `runs=20` 最终门槛评测。
3. 在 `RESULTS.md` 中补充修订后与 `Hybrid A*-MPC` 的门槛对比结论。
