# v1 - 变更

## 版本意图（版本级）
- Config: `repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1.json`
- Config: `repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke.json`

## 相对上一版的具体变更
- 该 strict no-fallback 版本链的初始版本。

## 代表性运行快照
- run_dir: `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017`
- run_json: `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017/configs/run.json`

## 关键参数快照（重点）
- `demo_mode`: `dqfd`
- `episodes`: `120`
- `max_steps`: `600`
- `save_ckpt`: `auto`
- `gamma`: `0.997`
- `learning_rate`: `0.0001`
- `batch_size`: `128`
- `learning_starts`: `300`
- `target_update_steps`: `2000`
- `target_update_tau`: `0.0`
- `eps_start`: `0.15`
- `eps_final`: `0.02`
- `eps_decay`: `8000`
- `dqfd_n_step`: `10`
- `dqfd_lambda_n`: `0.5`
- `dqfd_l2`: `1e-05`
- `demo_lambda`: `5.0`
- `demo_margin`: `2.0`
- `per_alpha`: `0.4`
- `per_beta0`: `0.6`
- `per_beta_steps`: `0`
- `per_eps_agent`: `0.001`
- `per_eps_demo`: `5.0`
- `forest_no_fallback`: `False`
- `forest_action_shield`: `True`
- `forest_adm_horizon`: `10`
- `forest_action_delta_dot_bins`: `7`
- `forest_action_accel_bins`: `7`
- `forest_random_start_goal`: `True`
- `forest_curriculum`: `True`
- `forest_rand_min_dist_m`: `6.0`
- `forest_rand_max_dist_m`: `0.0`
- `forest_rand_edge_margin_m`: `3.0`
- `forest_rand_fixed_prob`: `0.15`
- `forest_rand_tries`: `200`
- `forest_demo_prefill`: `True`
- `forest_demo_pretrain_steps`: `30000`
- `forest_demo_pretrain_early_stop_sr`: `0.75`
- `forest_demo_pretrain_early_stop_patience`: `2`
- `forest_demo_horizon`: `15`
- `replay_stratified`: `False`
- `replay_frac_demo`: `0.0`
- `replay_frac_goal`: `0.0`
- `replay_frac_stuck`: `0.0`
- `replay_frac_hazard`: `0.0`

## 命令覆盖（如有）
- argv: `train.py --profile repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke`
