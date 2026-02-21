# v7p2p8 改动清单（相对 v7p2p7）

## 变更目标
- 基于 `v7p2p7` 的“long 恢复但 short 退化”现象，执行一次大胆多变量训练稳定器改动，验证是否可同时稳住 short 并保持 long 恢复。

## 代码/配置改动明细

### 1) 多变量配置改动（核心）
- `configs/v7p2p8.json`
- `configs/repro_20260221_v7p2p8_bold_dynamic_expert_smoke.json`
- 关键变更（`old -> new`）：
  - `train.forest_train_dynamic_curriculum: false -> true`
  - `train.forest_expert_exploration: false -> true`
  - `train.forest_reward_no_progress_penalty: 0.35 -> 0.45`
  - `train.forest_train_dynamic_target_sr_short: (default) -> 0.75`
  - `train.forest_train_dynamic_target_sr_long: (default) -> 0.75`
  - `train.forest_train_dynamic_k: (default) -> 0.2`
  - `train.forest_train_dynamic_min_short_prob: (default) -> 0.25`
  - `train.forest_train_dynamic_max_short_prob: (default) -> 0.85`
  - `train.forest_expert_prob_start: (default) -> 0.35`
  - `train.forest_expert_prob_final: (default) -> 0.08`
  - `train.forest_expert_prob_decay: (default) -> 0.7`
  - `train.forest_expert_adapt_k: (default) -> 0.2`
  - `train.forest_expert_recent_window: (default) -> 30`

### 2) 保持不变（用于限定改动边界）
- `cnn_backbone=globalcnn_fusion`、`cnn_fusion_layernorm=true`
- `grad_clip_norm=10.0`
- `reward_scale=0.1`、`reward_clip_abs=10.0`
- `forest_demo_pretrain_min_effective_steps=8000`
- `forest_no_fallback=false`（`shielded/hybrid` 口径）

### 3) 运行与留档
- 新增 `docs/versions/v7p2p8/` 四件套并登记真实 run 路径、KPI 与失败分布。
- 同步更新版本总索引与仓库 README 中“最新训练/推理命令”。

## 受影响文件清单
- `configs/v7p2p8.json`
- `configs/repro_20260221_v7p2p8_bold_dynamic_expert_smoke.json`
- `docs/versions/v7p2p8/README.md`
- `docs/versions/v7p2p8/CHANGES.md`
- `docs/versions/v7p2p8/RESULTS.md`
- `docs/versions/v7p2p8/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
