# v7p2p7 改动清单（相对 v7p2p6）

## 变更目标
- 在不改算法主干与推理口径的前提下，验证 `v7p2p6` long 崩塌是否由过严梯度裁剪引起。
- 采用单变量策略：仅恢复 `grad_clip_norm`。

## 代码/配置改动明细

### 1) 单变量配置改动（核心）
- `configs/v7p2p7.json`
- `configs/repro_20260221_v7p2p7_gradclip_recover_smoke.json`
- 关键变更：
  - `train.grad_clip_norm: 5.0 -> 10.0`

### 2) 其余关键设置保持不变（用于隔离变量）
- 保持 `reward_scale=0.1`、`reward_clip_abs=10.0`。
- 保持 `forest_demo_pretrain_min_effective_steps=8000`。
- 保持 `cnn_backbone=globalcnn_fusion`、`cnn_fusion_layernorm=true`。
- 保持 `forest_no_fallback=false`（`shielded/hybrid` 口径）。

### 3) 运行与留档
- 新增 `docs/versions/v7p2p7/` 四件套并登记真实 run 路径、KPI 与失败分布。
- 同步更新版本总索引与仓库 README 中的“最新实验候选命令/版本索引”。

## 受影响文件清单
- `configs/v7p2p7.json`
- `configs/repro_20260221_v7p2p7_gradclip_recover_smoke.json`
- `docs/versions/v7p2p7/README.md`
- `docs/versions/v7p2p7/CHANGES.md`
- `docs/versions/v7p2p7/RESULTS.md`
- `docs/versions/v7p2p7/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
