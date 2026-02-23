# v8p6 变更清单（相对 v8p5）

## 1) 代码改动

### `forest_vehicle_dqn/cli/infer.py`
- 新增 `--forest-replace-topq`（替换动作候选 Top-Q 约束）：
  - 当进入替换逻辑时，先在 admissible candidates 中取 Top-Q（按 Q+turn penalty 排序），再做 `--forest-replace-ranking` tie-break。
  - 默认 `0`（不启用，保持 `v8p5` 行为）；`1` 退化为纯 Q 替换。

### `forest_vehicle_dqn/cli/train.py`
- 与 `infer.py` 保持一致：新增 `--forest-replace-topq`，并在训练期 inadmissible 替换逻辑与 train-progress suites 评测中一致使用。

### `tests/test_v8p6_replace_topq.py`
- 新增单测：验证 infer/train 两侧 replacement 选择在 `replace_topq` 下的一致性与退化行为（Top-1 等价 pure Q）。

## 2) 配置与文档

- 新增：
  - `configs/v8p6.json`
  - `configs/repro_20260223_v8p6_replace_topq_infer_smoke.json`
  - `docs/plans/2026-02-23-v8p6-replace-topq-design.md`
  - `docs/versions/v8p6/`（四件套）
- 更新：
  - `configs/INDEX.md`、`README.md`、`README.zh-CN.md`、`docs/versions/README.md`（索引与命令同步）

