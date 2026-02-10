# v3p4 - 变更

## 版本意图（版本级）
- 本版本在 `configs/` 下无独立配置文件；基于上一 profile 的 CLI 覆盖。

## 相对上一版的具体变更
- `episodes`: `120` -> `180`
- `save_ckpt`: `pretrain` -> `final`
- `eps_start`: `0.2` -> `0.3`
- `eps_decay`: `4500` -> `8000`
- `forest_action_delta_dot_bins`: `15` -> `7`
- `forest_action_accel_bins`: `15` -> `7`
- `forest_action_grid_power`: `2.0` -> `1.0`
- `forest_demo_pretrain_steps`: `30000` -> `20000`
- `forest_demo_pretrain_early_stop_sr`: `0.5` -> `0.7`
- `forest_demo_pretrain_early_stop_patience`: `1` -> `2`

## 代表性运行快照
- run_dir: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke/train_20260209_154029`
- run_json: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke/train_20260209_154029/configs/run.json`

## 关键参数快照（重点）
- `demo_mode`: `dqfd`
- `episodes`: `180`
- `max_steps`: `1200`
- `save_ckpt`: `final`
- `gamma`: `0.997`
- `learning_rate`: `0.0001`
- `batch_size`: `128`
- `learning_starts`: `500`
- `target_update_steps`: `2000`
- `target_update_tau`: `0.0`
- `eps_start`: `0.3`
- `eps_final`: `0.02`
- `eps_decay`: `8000`
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
- `forest_action_delta_dot_bins`: `7`
- `forest_action_accel_bins`: `7`
- `forest_action_grid_power`: `1.0`
- `forest_random_start_goal`: `True`
- `forest_curriculum`: `True`
- `forest_rand_min_dist_m`: `6.0`
- `forest_rand_max_dist_m`: `0.0`
- `forest_rand_edge_margin_m`: `3.0`
- `forest_rand_fixed_prob`: `0.1`
- `forest_rand_tries`: `200`
- `forest_demo_prefill`: `True`
- `forest_demo_pretrain_steps`: `20000`
- `forest_demo_pretrain_early_stop_sr`: `0.7`
- `forest_demo_pretrain_early_stop_patience`: `2`
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
- argv: `train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke --out repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke --episodes 180 --save-ckpt final --forest-action-delta-dot-bins 7 --forest-action-accel-bins 7 --eps-start 0.30 --eps-decay 8000 --forest-demo-pretrain-steps 20000 --forest-demo-pretrain-early-stop-sr 0.7 --forest-demo-pretrain-early-stop-patience 2`
