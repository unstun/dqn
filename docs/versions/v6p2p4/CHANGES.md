# v6p2p4 - 变更

## 版本意图
- 在 `v6p2p3` 训练/推理统一口径下，新增 `DDPG/SAC` 作为连续控制对比算法。

## 相对 v6p2p3 的代码/配置变更
- 新增连续算法模块：
  - `forest_vehicle_dqn/continuous_agents.py`
    - `ContinuousAgentConfig`
    - `DDPGAgent`
    - `SACAgent`
    - `ContinuousReplayBuffer`
- 训练扩展：
  - `forest_vehicle_dqn/cli/train.py`
    - `--rl-algos` 支持 `ddpg sac`
    - 新增 `--cont-*` 一组连续算法超参
    - 新增 `train_one_continuous(...)` 连续训练流程
- 推理扩展：
  - `forest_vehicle_dqn/cli/infer.py`
    - `--rl-algos` 支持 `ddpg sac`
    - 新增 `rollout_continuous_agent(...)`
    - 按算法分流加载 DQN/连续模型
    - 修复连续动作边界构造：仅在 `ddpg/sac` 分支访问 `env.model`，避免非连续场景误引用
- 基准脚本扩展：
  - `forest_vehicle_dqn/cli/benchmark.py`
    - 算法白名单/标签加入 `ddpg sac`
- 新增配置：
  - `configs/v6p2p4.json`

## 文档与归档变更
- 新增版本四件套：
  - `docs/versions/v6p2p4/README.md`
  - `docs/versions/v6p2p4/CHANGES.md`
  - `docs/versions/v6p2p4/RESULTS.md`
  - `docs/versions/v6p2p4/runs/README.md`
- 其他索引与 README：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`
