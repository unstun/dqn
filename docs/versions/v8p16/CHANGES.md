# v8p16 — CHANGES

相对上一版本链（`v8p15` / `v8p11` 训练权重体系），本版改动聚焦 **训练侧（B 线）**：给 forest 全局观测补全“到目标的 cost-to-go 距离场通道”。

## 1) 代码改动

- `forest_vehicle_dqn/env.py`
  - `AMRBicycleEnv.__init__(..., obs_include_progress_dist: bool = False)`（是否把 progress 距离场 map 拼到观测里）
  - 新增 `_update_obs_progress_flat()`（根据 `progress_dist_mode` / `progress_cost_*` 生成下采样距离场通道）
  - `reset()`：goal/距离场更新后同步刷新 `progress-dist` 观测通道
  - `observation_space`：支持 `10 + 2*N^2`（启用后）与原 `10 + 1*N^2`（默认关闭）两种维度
- `forest_vehicle_dqn/cli/train.py`
  - 新增 `--forest-obs-include-progress-dist/--no-forest-obs-include-progress-dist`（布尔开关，透传到 env）
- `forest_vehicle_dqn/cli/infer.py`
  - 新增 `--forest-obs-include-progress-dist/--no-forest-obs-include-progress-dist`（布尔开关，透传到 env）
- `forest_vehicle_dqn/networks.py`
  - `infer_flat_obs_cnn_layout(obs_dim)` 支持 `AMRBicycleEnv: 10 + 2*N^2` 的 CNN layout 推断

## 2) 配置改动

- 新增 `configs/v8p16.json`（train+infer 主入口）
- 新增本地 smoke 复现配置：
  - `configs/repro_20260224_v8p16_train_smoke.json`
  - `configs/repro_20260224_v8p16_infer_smoke_short.json`
  - `configs/repro_20260224_v8p16_infer_smoke_long.json`

## 3) 风险点与口径

- 观测维度变化：必须训练新权重；旧 checkpoint 与新观测不匹配会报错（或维度不一致）。
- 口径：推理仍为 `shielded/hybrid`（不可写成 `strict-argmax`）。
- 反作弊：不改 `goal_tolerance_m`（终点容差）及 stop 阈值。

