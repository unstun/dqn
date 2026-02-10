# v3p8 - 变更

## 版本意图
- 在 strict no-fallback 约束下，引入 short/long 联合 checkpoint 选择。

## 相对 v3p7 的具体变更
- `train.py --save-ckpt auto` 的选择逻辑切换为 short/long 联合打分：
  - 先比较 success（先联合成功数，再最小套件成功数），再比较 planning cost。
- 新增训练 CLI 参数用于联合 checkpoint 选择：
  - `--save-ckpt-joint-short-long`
  - `--save-ckpt-suite-runs`
  - `--save-ckpt-short-min-dist-m`, `--save-ckpt-short-max-dist-m`
  - `--save-ckpt-long-min-dist-m`, `--save-ckpt-long-max-dist-m`

## 变更文件
- `forest_vehicle_dqn/cli/train.py`
- `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p8_smoke.json`

## 参数快照（重点）
- `save_ckpt`: `auto`
- `save_ckpt_joint_short_long`: `true`
- `save_ckpt_suite_runs`: `10`
- `save_ckpt_short_min_dist_m`: `6.0`
- `save_ckpt_short_max_dist_m`: `14.0`
- `save_ckpt_long_min_dist_m`: `42.0`
- `save_ckpt_long_max_dist_m`: `0.0`
- `forest_no_fallback`: `true`
