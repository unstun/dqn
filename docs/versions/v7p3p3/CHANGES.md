# v7p3p3 改动清单（相对 v7p3p2）

## 变更目标
- 在保留 `v7p3p2` 的 turn-aware 替换结构前提下，调参降低 `timeout`，尝试恢复 short/mid `success_rate`。

## 代码/配置改动明细

### 1) 参数调整（训练/推理一致，核心）
- `forest_topk_turn_penalty=1.0 -> 0.3`
- `forest_min_progress_m=-0.01 -> 0.0`

### 2) 新增配置
- `configs/v7p3p3.json`
- `configs/repro_20260222_v7p3p3_infergate_smoke.json`

### 3) 版本归档与索引
- 新增并回填 `docs/versions/v7p3p3/` 四件套。
- 同步更新：
  - `docs/versions/README.md`
  - `README.md`
  - `README.zh-CN.md`
  - `configs/INDEX.md`

## 受影响文件清单
- `configs/v7p3p3.json`
- `configs/repro_20260222_v7p3p3_infergate_smoke.json`
- `docs/versions/v7p3p3/README.md`
- `docs/versions/v7p3p3/CHANGES.md`
- `docs/versions/v7p3p3/RESULTS.md`
- `docs/versions/v7p3p3/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
- `configs/INDEX.md`

