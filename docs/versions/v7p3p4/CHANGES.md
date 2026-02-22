# v7p3p4 改动清单（相对 v7p3p3）

## 变更目标
- 修复推理期 admissible gating 的兜底逻辑：当 “progress 可采纳动作集为空” 时不再回退到原始 `argmax(Q)`，而是回退到 collision-safe 动作集合，避免 `collision` 回潮并提升 `success_rate` 稳定性。

## 代码/配置改动明细

### 1) 推理期 safe fallback（核心）
- `forest_vehicle_dqn/cli/infer.py`：
  - 当 `argmax(Q)` inadmissible 且 top-k replacement 失败时：
    - 先尝试 progress-mask（`fallback_to_safe=false`）
    - 若为空，再回退 safe-mask（`fallback_to_safe=true`）
    - 若仍为空，调用 `_fallback_action_short_rollout(...)` 兜底
  - `fallback_rate` 统计修复：对“最终动作 != 原始 `argmax(Q)`” 的步数计数。

### 2) 训练侧 greedy 评估动作选择同步
- `forest_vehicle_dqn/cli/train.py`（`_forest_policy_action_from_q(...)`）：
  - 与推理侧一致：progress-mask 为空时回退 safe-mask，再不行用 `_fallback_action_short_rollout(...)`。

### 3) 单测补齐（逻辑覆盖）
- `tests/test_v7p3p2_turn_aware_topk.py`：
  - 新增用例覆盖 “progress mask 为空 → safe fallback mask 生效”
  - 新增用例覆盖 “全部 mask 为空 → short rollout fallback 生效”

### 4) 新增配置
- `configs/v7p3p4.json`（train+infer，默认保持 v7p3p2 参数族）
- `configs/repro_20260222_v7p3p4_safe_fallback_infer_smoke.json`（infer-only smoke，固定 v7p3p2 checkpoint 复评）

## 受影响文件清单
- `forest_vehicle_dqn/cli/infer.py`
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_v7p3p2_turn_aware_topk.py`
- `configs/v7p3p4.json`
- `configs/repro_20260222_v7p3p4_safe_fallback_infer_smoke.json`
- `docs/versions/v7p3p4/README.md`
- `docs/versions/v7p3p4/CHANGES.md`
- `docs/versions/v7p3p4/RESULTS.md`
- `docs/versions/v7p3p4/runs/README.md`
