# v6p2p5 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v6p2p4`
- 本版口径：`shielded/hybrid`（保留 `cnn-ddqn` 规则不变，仅连续算法单独演进）
- 状态：**代码已完成，已 self-check + quickcheck 推理（short, runs=1），待 smoke/full**

## 本版目标
- 实现 `cnn-ddqn` 与 `DDPG/SAC` 的参数与推理规则隔离，避免连续算法调参影响 `cnn-ddqn`。
- 在不改 `cnn-ddqn` 训练/推理逻辑的前提下，仅增强连续算法（`DDPG/SAC`）可学习性与可用性。
- 建立并持续维护三算法差异文档 `ALGO_DIFF.md`。

## 方法概要
- 连续训练隔离：新增 `--cont-learning-starts`（连续 warmup 起训步数）与 `--cont-bc-lambda`（连续 actor 的 demo BC 权重）。
- 连续推理隔离：新增 `--cont-forest-adm-horizon`、`--cont-forest-min-progress-m`、`--cont-forest-min-od-m`。
- 连续 fallback 改进：连续动作不可容许时，从“最近离散动作”改为“先最大化短视距进展，再按 L2 最近打破平局”。

## 关键命令
- 自检：
  - `conda run -n ros2py310 python train.py --profile v6p2p5 --self-check`
  - `conda run -n ros2py310 python infer.py --profile v6p2p5 --self-check`
- 训练/推理（待执行）：
  - `conda run -n ros2py310 python train.py --profile v6p2p5`
  - `conda run -n ros2py310 python infer.py --profile v6p2p5`

## 代表 run
- 训练：`N/A`（本轮未执行训练）
- 推理：`runs/v6p2p5_quickcheck/20260219_025226`

## 当前结论
- 本轮只完成代码与配置隔离改造，并通过 self-check。
- quickcheck（short, runs=1）下，`CNN-DDQN` 与 `Hybrid A*-MPC` 可达，`DDPG/SAC` 仍为 timeout。
- 可靠结论仍需补 smoke 与 full 评测后写入 `RESULTS.md`。

## 下一步
1. 先跑 smoke（短训练 + short/mid/long 小样本推理）验证 `DDPG/SAC` timeout 是否下降。
2. smoke 通过后再跑 300 轮 + short/mid/long 各 5 轮，更新 `RESULTS.md` 与 `runs/README.md`。
3. 若连续算法仍以 timeout 为主，再迭代 `cont_bc_lambda`、`cont_learning_starts` 与 `cont_*` shield 阈值。
