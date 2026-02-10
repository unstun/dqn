# v3p9 - 变更

## 版本意图
- 保留 v3p8 联合 checkpoint 选择，尝试仅参数级 anti-timeout 调整。

## 相对 v3p8 的具体变更
- `forest_reward_no_progress_penalty`: `0.35 -> 0.45`
- `forest_reward_no_progress_eps_m`: `0.03 -> 0.04`
- `forest_reward_idle_speed_m_s`: `0.12 -> 0.18`
- `forest_demo_filter_min_progress_ratio`: `0.05 -> 0.08`

## 变更文件
- `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p9_smoke.json`

## 参数快照（重点）
- `forest_no_fallback`: `true`
- `save_ckpt_joint_short_long`: `true`
- reward/filter 调整如上。
