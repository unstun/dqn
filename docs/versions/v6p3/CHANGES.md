# v6p3 - 变更

## 版本意图
- 将研究主线收敛到 `cnn-ddqn` 与 `Hybrid A*-MPC` 的对比，暂停连续算法分支。

## 相对 v6p2p5 的配置变更（old -> new）
- `configs/v6p2p5.json` -> `configs/v6p3.json`
- `train.rl_algos`: `["cnn-ddqn", "ddpg", "sac"]` -> `["cnn-ddqn"]`
- `infer.rl_algos`: `["cnn-ddqn", "ddpg", "sac"]` -> `["cnn-ddqn"]`
- `infer.baselines`: 保持 `["hybrid_astar_mpc"]`
- `train.out`: `v6p2p5` -> `v6p3`
- `infer.models/out`: `v6p2p5` -> `v6p3`
- 连续分支专用参数：`cont_*`（连续算法专用超参）从 `v6p3` 主配置中移除。

## 新增复现配置
- `configs/repro_20260219_v6p3_cnn_ddqn_vs_hybrid_astar_mpc.json`
  - 固化 `self-check/train/infer` 命令
  - 固化关键对比口径（RL 仅 `cnn-ddqn`，baseline 仅 `Hybrid A*-MPC`）

## 文档与索引同步
- 新增版本四件套：
  - `docs/versions/v6p3/README.md`
  - `docs/versions/v6p3/CHANGES.md`
  - `docs/versions/v6p3/RESULTS.md`
  - `docs/versions/v6p3/runs/README.md`
- 同步入口与索引：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`
