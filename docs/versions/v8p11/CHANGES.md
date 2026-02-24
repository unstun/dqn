# v8p11 变更清单（相对 v8p10）

## 1) 主要策略（训练优先）

- 推理侧口径不再作为主优化对象：继承 v8p10 的 `dijkstra8_nocorner + w_clearance=2.0 + progress_q replacement`。
- 训练侧强化：
  - 开启 `forest_expert_exploration`（专家混入行为策略）
  - 强化 DQfD demo 阶段（prefill/pretrain 预算与 margin loss 权重）
  - `forest_action_shield=true` 保持 train/infer 的 hybrid 口径一致

## 2) 配置与文档

- 新增 `configs/v8p11.json`（版本入口）
- 新增 `configs/repro_20260224_v8p11_train_smoke.json`（训练 smoke，可复现）
- 新增 `configs/repro_20260224_v8p11_infer_smoke_{short,long}.json`（推理 smoke，可复现）
- 新增 `docs/versions/v8p11/` 四件套（本文件为变更明细）

## 3) 受影响文件清单

- `configs/v8p11.json`
- `configs/repro_20260224_v8p11_train_smoke.json`
- `configs/repro_20260224_v8p11_infer_smoke_short.json`
- `configs/repro_20260224_v8p11_infer_smoke_long.json`
- `configs/INDEX.md`
- `docs/versions/v8p11/README.md`
- `docs/versions/v8p11/CHANGES.md`
- `docs/versions/v8p11/RESULTS.md`
- `docs/versions/v8p11/runs/README.md`

