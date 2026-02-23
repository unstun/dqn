# v8p5 变更清单（相对 v8p4）

## 1) 代码改动

### `forest_vehicle_dqn/cli/infer.py`
- 新增 `--forest-replace-ranking`（替换动作排序策略；仅在 `argmax(Q)` 不可行且 `forest_no_fallback=false` 时生效）：
  - `q`：纯 Q 排序（默认，保持旧行为）
  - `progress_clearance_q`：progress_dist → clearance(OD) → Q
  - `clearance_progress_q`：clearance(OD) → progress_dist → Q
- 将 top-k / mask 分支的“替换动作选择”统一收敛到 `_forest_choose_replacement_candidate(...)`，便于消融与口径一致性。

### `forest_vehicle_dqn/cli/train.py`
- 与 `infer.py` 保持一致：新增 `--forest-replace-ranking`，并在训练期的 inadmissible 替换逻辑与训练内评测（train-progress suites）中一致使用。

## 2) 配置与文档

- 新增：
  - `configs/v8p5.json`
  - `configs/repro_20260223_v8p5_replace_ranking_regression.json`
  - `configs/repro_20260223_v8p5_replace_ranking_infer_smoke.json`
  - `docs/plans/2026-02-23-v8p5-replace-ranking-design.md`
  - `docs/versions/v8p5/`（四件套）
- 更新：
  - `configs/INDEX.md`（V8 迭代入口切换到 `v8p5`）
  - `README.md`、`README.zh-CN.md`、`docs/versions/README.md`（索引与命令同步到 `v8p5`）
