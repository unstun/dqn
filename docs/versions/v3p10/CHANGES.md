# v3p10 - 变更

## 版本意图
- 做分布对齐：训练侧启用 short/long 双套件采样。

## 相对 v3p9 的具体变更
- 新增训练期双套件采样参数：
  - `forest_train_two_suites=true`
  - `forest_train_short_prob=0.5`
  - `forest_train_short_min_dist_m=6.0`
  - `forest_train_short_max_dist_m=14.0`
  - `forest_train_long_min_dist_m=42.0`
  - `forest_train_long_max_dist_m=0.0`
- 将双套件采样逻辑应用到训练 episode reset 及训练侧 random eval/demo 采样。

## 变更文件
- `forest_vehicle_dqn/cli/train.py`
- `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p10_smoke.json`

## 参数快照（重点）
- `forest_no_fallback`: `true`
- `save_ckpt_joint_short_long`: `true`
- `forest_train_two_suites`: `true`
- `forest_train_short_prob`: `0.5`
