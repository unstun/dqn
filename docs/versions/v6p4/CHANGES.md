# v6p4 - 变更

## 版本意图
- 在不改变算法类别（仍为 `cnn-ddqn`）的前提下，先修正训练调度与监督强度，避免配置层面偏差掩盖算法真实性能。

## 相对 v6p3 的配置变更（old -> new）
- `configs/v6p3.json` -> `configs/v6p4.json`
- `train.out`: `v6p3` -> `v6p4`
- `infer.models/out`: `v6p3` -> `v6p4`
- `train.episodes`（训练回合数）: `300` -> `3000`
- `train.eps_start`（初始探索率）: `0.2` -> `1.0`
- `train.eps_final`（末端探索率）: `0.02` -> `0.05`
- `train.eps_decay`（探索率衰减回合）: `4500` -> `0(auto)`
- `train.eps_decay_ratio`（探索率自动衰减比例）: `N/A` -> `0.67`（`eps_decay<=0` 时按 `round(episodes * ratio)` 计算）
- `train.learning_starts`（开始梯度更新前最小回放步数）: `500` -> `2000`
- `train.per_beta_steps`（PER 的 β 退火步数）: `0(auto)` -> `900000(显式)`
- `train.demo_lambda`（示范 margin loss 权重）: `8.0` -> `1.5`
- `train.demo_margin`（示范 margin 大小）: `2.0` -> `0.8`
- `train.forest_demo_pretrain_steps`（示范预训练步数）: `30000` -> `12000`
- `train.forest_demo_pretrain_early_stop_sr`（预训练早停 SR 阈值）: `0.5` -> `0.65`
- `train.forest_demo_pretrain_early_stop_patience`（预训练早停连续次数）: `1` -> `2`
- `train.obs_map_size`（训练观测地图边长）: `12(default)` -> `16`
- `infer.obs_map_size`（推理观测地图边长）: `12(default)` -> `16`
- `forest_vehicle_dqn/cli/train.py`：新增 `emit_train_config_sanity_warnings(...)`（训练启动期参数体检告警函数），用于提示 `eps_decay`/`eps_start`/`demo_lambda`/`obs_map_size` 的高风险配置。
- `forest_vehicle_dqn/cli/train.py`：新增 `eps_decay` 自适应解析逻辑（`eps_decay<=0` 时按 `episodes * eps_decay_ratio` 自动计算并打印生效值）。

## 新增复现配置
- `configs/repro_20260220_v6p4_cnn_ddqn_schedule_demo_balance.json`
  - 固化 `self-check -> smoke -> full` 两阶段流程命令
  - 固化关键超参与 `N/A` 占位 artifact 路径（待运行后回填）
- `configs/repro_20260220_v6p4_eps_decay_auto_ratio067.json`
  - 固化 `eps_decay=0` + `eps_decay_ratio=0.67` 的自适应口径
  - 固化 `episodes=3000` 下解析值 `resolved_eps_decay_at_3000=2010`

## 文档与索引同步
- 新增版本四件套：
  - `docs/versions/v6p4/README.md`
  - `docs/versions/v6p4/CHANGES.md`
  - `docs/versions/v6p4/RESULTS.md`
  - `docs/versions/v6p4/runs/README.md`
- 代码改动：
  - `forest_vehicle_dqn/cli/train.py`
- 将同步更新：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`
