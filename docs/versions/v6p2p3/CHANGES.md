# v6p2p3 - 变更

## 版本意图
- 将 `v6p2p2` 中训练/推理规则不一致项统一到同一套 `hybrid/shielded` 口径，并固定奖励参数为 `k_t=0.10`、`k_delta=0.8`。

## 相对 v6p2p2 的代码/配置变更
- 训练参数新增：
  - `forest_vehicle_dqn/cli/train.py`
    - 新增 `--forest-topk`（训练动作替换时的 top-k）
    - 新增 `--forest-min-od-m`（训练 admissibility 最小净空阈值）
- 训练规则对齐：
  - `forest_vehicle_dqn/cli/train.py`
    - 新增 `forest_stop_action(...)`（停车覆盖动作）
    - 新增 `_forest_policy_action_from_q(...)`（infer 同口径动作选择：stop override + top-k/mask）
    - 训练进度评估 `_eval_train_progress_suites(...)` 改为调用统一动作规则
    - 训练/预训练/评估路径中的 `min_od_m=0.0` 硬编码改为可配置 `forest_min_od_m`
- 新增版本配置：
  - `configs/v6p2p3.json`
    - `forest_reward_k_t=0.10`
    - `forest_reward_k_delta=0.8`
    - 训练/推理统一：`forest_no_fallback=false`、`forest_adm_horizon=30`、`forest_min_progress_m=0.01`、`forest_min_od_m=0.02`、`forest_topk=10`、`no_terminate_on_stuck=true`

## 文档与归档变更
- 新增版本四件套：
  - `docs/versions/v6p2p3/README.md`
  - `docs/versions/v6p2p3/CHANGES.md`
  - `docs/versions/v6p2p3/RESULTS.md`
  - `docs/versions/v6p2p3/runs/README.md`
- 索引同步：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`
