# v3p11 - 变更

## 版本意图
- 恢复 v3p10 中 long 套件的 超时崩塌。

## 相对 v3p10 的具体变更
- 在训练双套件采样中加入 short-prob ramp：
  - `forest_train_short_prob`: `0.5 -> 0.35`
  - `forest_train_short_prob_ramp`: `0.0 -> 0.4`

## 变更文件
- `forest_vehicle_dqn/cli/train.py`
- `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke.json`

## 参数快照（重点）
- `forest_no_fallback`: `true`
- `save_ckpt_joint_short_long`: `true`
- `forest_train_two_suites`: `true`
- `forest_train_short_prob`: `0.35`
- `forest_train_short_prob_ramp`: `0.4`
