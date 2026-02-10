# v3p7 - 变更

## 版本意图（版本级）
- 本版本在 `configs/` 下无独立配置文件；基于上一 profile 的 CLI 覆盖。

## 相对上一版的具体变更
- `demo_mode`: `dqfd` -> `legacy`
- `episodes`: `180` -> `140`
- `save_ckpt`: `best` -> `final`
- `demo_lambda`: `8.0` -> `4.0`
- `demo_margin`: `2.0` -> `1.0`
- `replay_stratified`: `False` -> `True`
- `replay_frac_demo`: `0.0` -> `0.5`
- `replay_frac_goal`: `0.0` -> `0.2`
- `replay_frac_stuck`: `0.0` -> `0.15`
- `replay_frac_hazard`: `0.0` -> `0.15`

## 代表性运行快照
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p7_legacy_smoke/train_20260209_170153`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p7_legacy_smoke/train_20260209_170153/configs/run.json`

## 关键参数快照（重点）
- `demo_mode`: `legacy`
- `episodes`: `140`
- `max_steps`: `1200`
- `save_ckpt`: `final`
- `gamma`: `0.997`
- `learning_rate`: `0.0001`
- `batch_size`: `128`
- `learning_starts`: `500`
- `target_update_steps`: `2000`
- `target_update_tau`: `0.0`
- `eps_start`: `0.2`
- `eps_final`: `0.02`
- `eps_decay`: `4500`
- `dqfd_n_step`: `10`
- `dqfd_lambda_n`: `0.5`
- `dqfd_l2`: `1e-05`
- `demo_lambda`: `4.0`
- `demo_margin`: `1.0`
- `per_alpha`: `0.4`
- `per_beta0`: `0.6`
- `per_beta_steps`: `0`
- `per_eps_agent`: `0.001`
- `per_eps_demo`: `3.0`
- `forest_no_fallback`: `True`
- `forest_action_shield`: `False`
- `forest_adm_horizon`: `20`
- `forest_min_progress_m`: `-0.02`
- `forest_action_delta_dot_bins`: `15`
- `forest_action_accel_bins`: `15`
- `forest_action_grid_power`: `1.0`
- `forest_random_start_goal`: `True`
- `forest_curriculum`: `True`
- `forest_rand_min_dist_m`: `6.0`
- `forest_rand_max_dist_m`: `0.0`
- `forest_rand_edge_margin_m`: `3.0`
- `forest_rand_fixed_prob`: `0.1`
- `forest_rand_tries`: `200`
- `forest_demo_prefill`: `True`
- `forest_demo_pretrain_steps`: `30000`
- `forest_demo_pretrain_early_stop_sr`: `0.5`
- `forest_demo_pretrain_early_stop_patience`: `1`
- `forest_demo_horizon`: `3`
- `forest_demo_filter_min_progress_ratio`: `0.05`
- `forest_demo_filter_min_progress_per_step_m`: `0.0`
- `forest_demo_filter_max_steps`: `0`
- `forest_reward_no_progress_penalty`: `0.35`
- `forest_reward_no_progress_eps_m`: `0.03`
- `forest_reward_idle_speed_m_s`: `0.12`
- `forest_goal_admissible_relax_factor`: `2.5`
- `replay_stratified`: `True`
- `replay_frac_demo`: `0.5`
- `replay_frac_goal`: `0.2`
- `replay_frac_stuck`: `0.15`
- `replay_frac_hazard`: `0.15`

## 命令覆盖（如有）
- argv: `train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p7_legacy_smoke --demo-mode legacy --episodes 140 --save-ckpt final --replay-stratified --replay-frac-demo 0.5 --replay-frac-goal 0.2 --replay-frac-stuck 0.15 --replay-frac-hazard 0.15 --demo-lambda 4.0 --demo-margin 1.0`
