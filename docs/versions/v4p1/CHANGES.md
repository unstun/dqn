# v4p1 - 变更

## 版本意图
- 在不破坏 `DDQN/DQfD` 学术定义和 strict no-fallback 推理约束的前提下，提高长程稳定性并减少 collision 主导退化。

## 相对上一版（v3p12 / v3p12recover）的具体改动

### 1) 动态 short/long 课程调度（训练期）
- `forest_train_short_prob`：由固定值/线性 ramp 扩展为可选动态更新。
- 新增逻辑：基于 train-progress 的 `sr_short/sr_long` 与目标 SR 偏差更新 `p_short`。
- 新增参数：
  - `--forest-train-dynamic-curriculum`
  - `--forest-train-dynamic-target-sr-short`
  - `--forest-train-dynamic-target-sr-long`
  - `--forest-train-dynamic-k`
  - `--forest-train-dynamic-min-short-prob`
  - `--forest-train-dynamic-max-short-prob`

### 2) failure-aware PER priority boost（训练期）
- 在 PER priority 更新中加入 flag 相关 boost（不改 TD 定义）：
  - `near_goal`、`stuck`、`hazard`。
- 新增参数：
  - `--per-boost-near-goal`
  - `--per-boost-stuck`
  - `--per-boost-hazard`

### 3) checkpoint 选择加入 long SR 下限
- 在 `save_ckpt_joint_short_long` 比较中，先比较是否满足 `long_success_rate >= floor`，再比较原有 joint 成功与代价。
- 新增参数：
  - `--save-ckpt-long-sr-floor`

### 4) 训练可观测性增强
- train-progress 新增 `argmax_inadmissible_rate_short/long/all` 统计，便于判断策略是否在收敛到不可行动作区域。

## 变更文件
- `forest_vehicle_dqn/cli/train.py`
- `forest_vehicle_dqn/agents.py`
- `forest_vehicle_dqn/replay_buffer.py`
- `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke.json`

## 关键参数快照（v4p1 smoke）
- `forest_no_fallback`: `true`
- `forest_train_two_suites`: `true`
- `forest_train_dynamic_curriculum`: `true`
- `forest_train_dynamic_target_sr_short`: `0.7`
- `forest_train_dynamic_target_sr_long`: `0.7`
- `forest_train_dynamic_k`: `0.2`
- `per_boost_near_goal`: `0.1`
- `per_boost_stuck`: `0.25`
- `per_boost_hazard`: `0.35`
- `save_ckpt_long_sr_floor`: `0.2`

## 学术定义与 strict no-fallback 合规说明
- `DDQN`：仍为 online `argmax` + target eval 的 Bellman 目标。
- `DQfD`：仍为 PER + 1-step/n-step TD + margin (+L2) 的定义口径。
- strict no-fallback：推理期仍保持纯 `argmax(Q)`，未新增替代决策路径。
