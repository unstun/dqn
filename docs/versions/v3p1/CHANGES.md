# v3p1 - 变更

## 版本意图（版本级）
- Config: `repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1.json`
- Config: `repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke.json`

## 相对上一版的具体变更
- `episodes`: `120` -> `300`
- `eps_decay`: `4500` -> `9000`
- `forest_min_progress_m`: `0.0` -> `-0.02`
- `forest_action_grid_power`: `<unset>` -> `1.0`
- `forest_demo_pretrain_steps`: `24000` -> `40000`
- `forest_demo_pretrain_early_stop_sr`: `<unset>` -> `0.5`
- `forest_demo_pretrain_early_stop_patience`: `<unset>` -> `1`
- `forest_demo_filter_min_progress_ratio`: `0.2` -> `0.05`
- `forest_demo_filter_min_progress_per_step_m`: `0.003` -> `0.0`
- `forest_demo_filter_max_steps`: `320` -> `0`
- `replay_stratified`: `<unset>` -> `False`
- `replay_frac_demo`: `<unset>` -> `0.0`
- `replay_frac_goal`: `<unset>` -> `0.0`
- `replay_frac_stuck`: `<unset>` -> `0.0`
- `replay_frac_hazard`: `<unset>` -> `0.0`

## 代表性运行快照
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt/train_20260209_143827`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt/train_20260209_143827/configs/run.json`

## 关键参数快照（重点）
- `demo_mode`: `dqfd`
- `episodes`: `300`
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
- `eps_decay`: `9000`
- `dqfd_n_step`: `10`
- `dqfd_lambda_n`: `0.5`
- `dqfd_l2`: `1e-05`
- `demo_lambda`: `8.0`
- `demo_margin`: `2.0`
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
- `forest_demo_pretrain_steps`: `40000`
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
- `replay_stratified`: `False`
- `replay_frac_demo`: `0.0`
- `replay_frac_goal`: `0.0`
- `replay_frac_stuck`: `0.0`
- `replay_frac_hazard`: `0.0`

## 命令覆盖（如有）
- argv: `train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1 --save-ckpt final --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt`
