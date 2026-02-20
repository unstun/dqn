# v7p2p1 - 变更

## 版本意图
- 归档 `v7p2` 本轮收益不稳定的实验结果。
- 将主线实现与默认命令回退到 `v7p1`，用于后续稳定迭代。

## 相对 v7p2 的代码/配置变更
- 代码回退（核心）：
  - `forest_vehicle_dqn/env.py`
    - `AMRBicycleEnv` 观测从 `11 + N^2` 回退到 `10 + N^2`。
    - 移除 `prev_a_n` 标量通道，恢复 `v7p1` 观测布局。
  - `forest_vehicle_dqn/networks.py`
    - `infer_flat_obs_cnn_layout(...)` bicycle 识别从 `11 + N^2` 回退到 `10 + N^2`。
- 测试回退：
  - 删除 `tests/test_v7p2_markov_obs_prev_a.py`（该测试只适用于 `v7p2` 的 11 维标量观测）。

## 文档与流程变更
- README 与版本索引更新：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`
  - 新增 `v7p2p1` 版本索引项，并将主线推荐 profile 回退为 `v7p1`。
- 标准工作流写入规范：
  - `AGENTS.md`
  - 新增“版本标准工作流（pre-push -> smoke150/runs=3 -> go/no-go -> rollback -> 归档）”。

## 新增留档与复现配置
- 新增版本四件套：
  - `docs/versions/v7p2p1/README.md`
  - `docs/versions/v7p2p1/CHANGES.md`
  - `docs/versions/v7p2p1/RESULTS.md`
  - `docs/versions/v7p2p1/runs/README.md`
- 新增复现配置：
  - `configs/repro_20260220_v7p2p1_rollback_v7p1.json`
