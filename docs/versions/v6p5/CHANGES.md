# v6p5 - 变更

## 版本意图
- 将 `AMRBicycleEnv`（自行车模型环境）的观测从“整图低分辨率”升级为“局部高分辨率”，优先修复观测分辨率瓶颈。

## 相对 v6p4 的配置变更（old -> new）
- `configs/v6p4.json` -> `configs/v6p5.json`
- `train.out`: `v6p4` -> `v6p5`
- `infer.models/out`: `v6p4` -> `v6p5`
- `train.obs_map_mode`（训练观测模式）: `global(default)` -> `local`
- `infer.obs_map_mode`（推理观测模式）: `global(default)` -> `local`
- `train.obs_map_size`（训练观测边长）: `16` -> `64`
- `infer.obs_map_size`（推理观测边长）: `16` -> `64`
- `train.obs_local_range_m`（训练局部窗口半径）: `N/A` -> `18.0`
- `infer.obs_local_range_m`（推理局部窗口半径）: `N/A` -> `18.0`

## 后续补充（CUDA 优先 profile）
- 新增 `configs/v6p5_cuda_pref.json`（同方法族的 GPU 吞吐优先配置）：
  - `train.batch_size`: `128` -> `256`
  - `train.train_freq`（每隔多少环境步触发一次更新）: `4(default)` -> `1`
  - `train.train_eval_every`（训练过程短评估间隔）: `10(default)` -> `0`
  - `train.device`: `cuda`（保持）
  - `infer.device`: `auto` -> `cuda`
  - `train/infer.out/models`: `v6p5` -> `v6p5_cuda_pref`

## 代码改动
- `forest_vehicle_dqn/networks.py`
  - `CNNQNetwork`（CNN 版 Q 网络）卷积骨干由 `Conv2d -> ReLU` 调整为 `Conv2d -> BatchNorm2d -> ReLU`（三层），用于稳定训练期特征分布。
  - 保持输入输出维度不变；注意旧 checkpoint 与新结构不兼容，需重训。
- `forest_vehicle_dqn/env.py`
  - `AMRBicycleEnv.__init__(...)` 增加 `obs_map_mode` 与 `obs_local_range_m`。
  - 新增 `_observe_occ_flat()`（按模式生成 occupancy 观测）；`local` 模式下每步裁剪局部 patch。
  - `_observe()` 改为调用 `_observe_occ_flat()`。
- `forest_vehicle_dqn/cli/train.py`
  - 新增参数：`--obs-map-mode`、`--obs-local-range-m`。
  - 创建 `AMRBicycleEnv` 时透传本参数。
- `forest_vehicle_dqn/cli/infer.py`
  - 新增参数：`--obs-map-mode`、`--obs-local-range-m`。
  - 创建 `AMRBicycleEnv` 时透传本参数。

## 新增复现配置
- `configs/repro_20260220_v6p5_local_obs_mid360_nearfull.json`
  - 固化 `local` 观测口径（`obs_map_mode=local`, `obs_map_size=64`, `obs_local_range_m=18.0`）
  - 固化 `self-check -> smoke -> full` 命令
- `configs/repro_20260220_v6p5_cnn_batchnorm.json`
  - 在 v6p5 口径上新增 `CNNQNetwork` 的 BatchNorm 复现命令与关键超参记录
- `configs/repro_20260220_v6p5_cuda_pref_gpu_throughput.json`
  - 固化 `v6p5_cuda_pref` 的 `self-check -> smoke -> full` 命令与关键超参记录

## 文档与索引同步
- 新增版本四件套：
  - `docs/versions/v6p5/README.md`
  - `docs/versions/v6p5/CHANGES.md`
  - `docs/versions/v6p5/RESULTS.md`
  - `docs/versions/v6p5/runs/README.md`
- 将同步更新：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`
