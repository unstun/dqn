# v7p2p9 改动清单（相对 v7p2p8）

## 变更目标
- 对 `v7p2p8` 的多变量方案进行分解消融，优先验证 `expert exploration` 对 short/long 波动的贡献。

## 代码/配置改动明细

### 1) 单变量配置改动（核心）
- `configs/v7p2p9.json`
- `configs/repro_20260221_v7p2p9_ablate_expert_smoke.json`
- 关键变更：
  - `train.forest_expert_exploration: true -> false`

### 2) 保持不变（用于隔离变量）
- `train.forest_train_dynamic_curriculum=true`
- `train.forest_reward_no_progress_penalty=0.45`
- `train.grad_clip_norm=10.0`
- `train.reward_scale=0.1`
- `train.reward_clip_abs=10.0`
- `train.forest_demo_pretrain_min_effective_steps=8000`
- `forest_no_fallback=false`（`shielded/hybrid` 口径）

### 3) 运行与留档
- 新增 `docs/versions/v7p2p9/` 四件套并登记真实 run 路径、KPI 与失败分布。
- 同步更新版本总索引与仓库 README 中“最新训练/推理命令”。

## 受影响文件清单
- `configs/v7p2p9.json`
- `configs/repro_20260221_v7p2p9_ablate_expert_smoke.json`
- `docs/versions/v7p2p9/README.md`
- `docs/versions/v7p2p9/CHANGES.md`
- `docs/versions/v7p2p9/RESULTS.md`
- `docs/versions/v7p2p9/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
