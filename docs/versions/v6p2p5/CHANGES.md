# v6p2p5 - 变更

## 版本意图
- 隔离 `cnn-ddqn` 与 `DDPG/SAC` 的训练/推理调参通道，确保连续算法迭代不改 `cnn-ddqn` 行为口径。

## 相对 v6p2p4 的代码/配置变更
- 连续算法配置与训练（仅 DDPG/SAC 生效）：
  - `forest_vehicle_dqn/continuous_agents.py`
    - `ContinuousAgentConfig` 新增 `cont_bc_lambda`。
    - `ContinuousReplayBuffer` 支持返回 demo mask（用于 actor 端 BC 正则）。
    - `DDPGAgent.update()` / `SACAgent.update()` 新增 demo BC 正则项（受 `cont_bc_lambda` 控制，默认 `0` 关闭）。
  - `forest_vehicle_dqn/cli/train.py`
    - 新增 `--cont-learning-starts`（连续 warmup 步数，默认回落到 `--learning-starts`）。
    - 新增 `--cont-bc-lambda`（连续 actor BC 权重）。
    - 连续 demo target 计算使用连续分支的 learning_starts（不再强绑定 DQN）。

- 连续推理隔离（仅 DDPG/SAC 生效）：
  - `forest_vehicle_dqn/cli/infer.py`
    - 新增 `--cont-forest-adm-horizon` / `--cont-forest-min-progress-m` / `--cont-forest-min-od-m`。
    - 连续 fallback 从“最近可容许离散动作”升级为“进展优先 + 距离次级”。

- 新增配置：
  - `configs/v6p2p5.json`
  - `configs/repro_20260219_v6p2p5_cont_isolation.json`

## 文档与归档变更
- 新增版本四件套：
  - `docs/versions/v6p2p5/README.md`
  - `docs/versions/v6p2p5/CHANGES.md`
  - `docs/versions/v6p2p5/RESULTS.md`
  - `docs/versions/v6p2p5/runs/README.md`
- 新增三算法差异文档：
  - `docs/versions/v6p2p5/ALGO_DIFF.md`
- 同步索引与命令：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`
